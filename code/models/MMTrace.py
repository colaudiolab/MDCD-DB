import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math

class Patches3D(nn.Module):
    def __init__(self, patch_size, patch_depth):
        super(Patches3D, self).__init__()
        self.patch_size = patch_size
        self.patch_depth = patch_depth

    def forward(self, volume):
        batch_size = volume.size(0)
        patches = F.unfold(volume, kernel_size=(self.patch_depth, self.patch_size, self.patch_size),
                           stride=(self.patch_depth, self.patch_size, self.patch_size))
        patches = rearrange(patches, 'b c (t h w) -> b (t h w) c', t=self.patch_depth, h=self.patch_size, w=self.patch_size)
        return patches

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, filters, conv_num=3, activation="relu"):
        super(ResidualBlock, self).__init__()
        self.conv_num = conv_num
        self.activation = nn.ReLU() if activation == "relu" else None
        
        # Shortcut
        self.shortcut = nn.Conv1d(in_channels=in_channels, out_channels=filters, kernel_size=1, padding='same')
        
        # Convolutional layers
        self.conv_layers = nn.ModuleList()
        for i in range(conv_num - 1):
            if i== 0:
                self.conv_layers.append(nn.Conv1d(in_channels=in_channels, out_channels=filters, kernel_size=3, padding='same'))
            else:
                self.conv_layers.append(nn.Conv1d(in_channels=filters, out_channels=filters, kernel_size=3, padding='same'))
            self.conv_layers.append(nn.ReLU())
        self.conv_layers.append(nn.Conv1d(in_channels=filters, out_channels=filters, kernel_size=3, padding='same'))
        
        # Max pooling
        self.max_pool = nn.MaxPool1d(kernel_size=2, stride=2)
    
    def forward(self, x):
        s = self.shortcut(x)
        # Applying convolutional layers
        for i, layer in enumerate(self.conv_layers):
            if isinstance(layer, nn.Conv1d):
                # Applying kernel and activity regularization manually
                x = layer(x)
                x = x + torch.norm(layer.weight, p=1)
                x = x + torch.norm(x, p=2)
                # x = torch.norm(x, p=2)
            else:
                x = layer(x)
        
        # Adding shortcut
        x = x + s
        x = self.activation(x)
        
        # Max pooling
        x = self.max_pool(x)
        return x

class AudioModel(nn.Module):
  def __init__(self):
    super(AudioModel, self).__init__()
    self.residual_blocks1 = ResidualBlock(1, 64, 2)
    self.residual_blocks2 = ResidualBlock(64,64, 2)
    self.residual_blocks3 = ResidualBlock(64,128, 3)
    self.residual_blocks4 = ResidualBlock(128,256, 3)
    self.residual_blocks5 = ResidualBlock(256,384, 3)

    self.batch_norm1 = nn.BatchNorm1d(64, eps=1e-3)
    self.batch_norm2 = nn.BatchNorm1d(64, eps=1e-3)
    self.batch_norm3 = nn.BatchNorm1d(128, eps=1e-3)
    self.batch_norm4 = nn.BatchNorm1d(256, eps=1e-3)
    self.batch_norm5 = nn.BatchNorm1d(384, eps=1e-3)

  def forward(self, x):
    nb_samp = x.shape[0]
    len_seq = x.shape[1]
    x = x.view(nb_samp, 1, len_seq)
    x = self.residual_blocks1(x)
    x = self.batch_norm1(x)
    x = self.residual_blocks2(x)
    x = self.batch_norm2(x)
    x = self.residual_blocks3(x)
    x = self.batch_norm3(x)
    x = self.residual_blocks4(x)
    x = self.batch_norm4(x)
    x = self.residual_blocks5(x)
    x = self.batch_norm5(x)
    return x
    

class AudioFFTLayer(nn.Module):
    def __init__(self):
        super(AudioFFTLayer, self).__init__()
    
    def forward(self, audio):
        fft = torch.fft.fft(
            torch.complex(real=audio, imag=torch.zeros_like(audio))
        )
        fft = torch.unsqueeze(fft, dim=-1)

        # Return the absolute value of the first half of the FFT
        # which represents the positive frequencies
        out = torch.abs(fft[:, : (audio.size(1) // 2), :])
        return out

class MLPMixerLayer(nn.Module):
    def __init__(self, input_shape, num_patches, hidden_units, dropout_rate):
        super(MLPMixerLayer, self).__init__()

        self.mlp1 = nn.Sequential(
            nn.Linear(num_patches, num_patches),
            nn.GELU(),
            nn.Linear(num_patches, num_patches),
            nn.Dropout(p=dropout_rate),
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(hidden_units, hidden_units),
            nn.GELU(),
            nn.Linear(hidden_units, hidden_units),
            nn.Dropout(p=dropout_rate),
        )
        self.normalize = nn.LayerNorm(normalized_shape=hidden_units, eps=1e-6)

    def forward(self, inputs):
        x = self.normalize(inputs)
        x_channels = torch.permute(x, (0, 2, 1))
        mlp1_outputs = self.mlp1(x_channels)
        mlp1_outputs = torch.permute(mlp1_outputs, (0, 2, 1))
        x = mlp1_outputs + inputs
        mlp2_outputs = self.mlp2(x)
        x = mlp2_outputs + inputs
        x = self.normalize(x)
        return x

class MMTrace(nn.Module):
    def __init__(self, embedding_dim=384, num_patches=1000, num_patches_audio=1000, dropout_rate=0.2, positional_encoding=False):
        super(MMTrace, self).__init__()
        self.num_patches = num_patches
        self.num_patches_audio = num_patches_audio
        self.audio_fft_layer = AudioFFTLayer()
        self.audio_model = AudioModel()
        self.out_dim = embedding_dim
        self.fmap_dim = embedding_dim
        num_blocks = 1
        # self.video_resnet = torchvision.models.video.r3d_18(pretrained=False)
        # self.video_resnet.stem[0] = nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
        # self.video_resnet.fc = nn.Linear(self.video_resnet.fc.in_features, 10)
        self.video_resnet = ResNet3D(BasicBlock3D, [2, 2, 2, 2], (40,5,5))

        self.video_linear = nn.Linear(384, embedding_dim)
        self.positional_encoding = positional_encoding
        if positional_encoding:
            self.position_embedding_video = nn.Embedding(num_embeddings=num_patches, embedding_dim=embedding_dim)
            self.position_embedding_audio = nn.Embedding(num_embeddings=num_patches_audio, embedding_dim=embedding_dim)
        self.mlp_mixer_video = MLPMixerLayer(embedding_dim, num_patches, embedding_dim, dropout_rate)
        self.mlp_mixer_audio = MLPMixerLayer(embedding_dim, num_patches_audio, embedding_dim, dropout_rate)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.mlp_mixer_blocks = nn.ModuleList([MLPMixerLayer(embedding_dim, num_patches + num_patches_audio, embedding_dim, dropout_rate) for _ in range(num_blocks)])
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.logits = nn.Linear(embedding_dim, 2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                # m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                m.bias.data.zero_()

    def video_processing(self, model, inp):

        # inp = inp.view((inp.size()[0], 10, -1,) + inp.size()[2:])
        # x = []
        # for i in range(10):
        #     tmp_inp = inp[:, i, :, :, :].contiguous()
        #     tmp_inp = tmp_inp.view(tmp_inp.size()[0], -1, 3, tmp_inp.size()[2], tmp_inp.size()[3]).permute(0, 2, 1, 3, 4).contiguous()
        #     # x.append(model(inp[:, i, :, :, :].contiguous()))
        #     x.append(model(tmp_inp.contiguous()))
        # x = torch.stack(x, dim=1)
        # return x
        inp = inp.view((inp.size()[0], 3, -1,) + inp.size()[2:])
        x = model(inp.contiguous())
        return x

    def forward(self, inputs_video, inputs_audio):
        audio_fft = self.audio_fft_layer(inputs_audio)
        print('1',audio_fft[0])
        x_audio = self.audio_model(audio_fft)
        x_audio = x_audio.permute(0, 2, 1)
        
        patches_video = self.video_processing(self.video_resnet, inputs_video)
        # patches_video = self.video_resnet(inputs_video)
        patches_video = torch.reshape(patches_video, (patches_video.shape[0], -1, 384))
        x_video = self.video_linear(patches_video)

        if self.positional_encoding:
            positions_video = torch.arange(start=0, end=self.num_patches, device=inputs_video.device)
            position_embedding_video = self.position_embedding_video(positions_video)
            x_video = x_video + position_embedding_video

            positions_audio = torch.arange(start=0, end=self.num_patches_audio, device=inputs_audio.device)
            position_embedding_audio = self.position_embedding_audio(positions_audio)
            x_audio = x_audio + position_embedding_audio

        print('a',x_audio[0][0])
        x_video = self.mlp_mixer_video(x_video)
        x_audio = self.mlp_mixer_audio(x_audio)
        
        x = torch.cat((x_video, x_audio), dim=1)
        x = self.layer_norm(x)
        for mlp_mixer_block in self.mlp_mixer_blocks:
            x = mlp_mixer_block(x)
        # (batch_size, 2000, 384)
        fusion_out = self.avg_pool(x.transpose(1, 2)).squeeze(1).squeeze(2)
        # (batch_size, 384)
        fusion_out = self.dropout(fusion_out)
        # logits = self.logits(fusion_out)
        # return torch.sigmoid(logits)
        out = {'video': x_video, 'audio': x_audio, 'features': fusion_out, 'fmaps': x}
        return out
        # return x_video, x_audio, fusion_out

# Note: PyTorch's nn module does not have GCAdamW or get_gradients method like TensorFlow's tfa.optimizers.AdamW.
# You would need to define your own optimizer or modify the existing ones if you want to achieve similar functionality.

class GCAdamW(torch.optim.AdamW):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01):
        super(GCAdamW, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.dim() > 1:
                    grad -= torch.mean(grad, dim=list(range(grad.dim()-1)), keepdim=True)
                p.data.add_(grad, alpha=-group['lr'] * (1 + group['weight_decay']))

        return loss

class BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(
                    in_channels,
                    self.expansion * out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm3d(self.expansion * out_channels),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet3D(nn.Module):
    def __init__(self, block, layers, target_shape=(5,5,5)):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv3d(
            3, 64, kernel_size=(3,7,7), stride=(1,2,2), padding=(1,3,3), bias=False
        )
        self.bn1 = nn.BatchNorm3d(64)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.adaptive_pool = nn.AdaptiveAvgPool3d(target_shape)
        self.final_conv = nn.Conv3d(512, 384, kernel_size=1)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.adaptive_pool(x)  # (B, 512, 5, 5, 5)
        x = self.final_conv(x)     # (B, 384, 5, 5, 5)
        return x

if __name__ == "__main__":
    net = MMTrace()
    # print(net)
    # print(sum(p.numel() for p in net.parameters() if p.requires_grad))
    # import torchvision
    # net2 = torchvision.models.resnet18()
    # print(sum(p.numel() for p in net2.parameters() if p.requires_grad))
    # y = net(torch.randn(1, 120, 128, 128), torch.randn(1, 64000))
    y = net(torch.randn(2, 120, 128, 128), torch.randn(2, 64000))
    x_video, x_audio, x, _ = y['video'], y['audio'], y['features'], y['fmaps']
    print(x_video, x_audio, x)
    # print(x_video.shape, x_audio.shape, x.shape)
    # print(summary(net, (10, 512)))