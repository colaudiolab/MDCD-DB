import torch
import torch.nn.functional as F
import numpy as np

from functools import partial


class MRFA:
    def __init__(self, with_input_norm=True) -> None:
        self._init_inbatch_properties()
        self.with_input_norm = with_input_norm

        self.perturbations = []
        self.remove_handles = []
    
    def _init_inbatch_properties(self):
        self.perturbation_layers = []
        self.perturbation_factor = []
        self.perturbation_idices = [] # perturbation indices in dataset
        self.perturbation_idices_inbatch = [] # perturbation indices in batch

    def feature_augmentation(self, model, video, audio, targets, net_type):
        if net_type == 'GAT':
            if isinstance(model, torch.nn.DataParallel):
                self.get_feature_augmentation(model.module, model.module.backbone, video, audio, targets, 4, register_forward_prehook_GAT)
            else:
                self.get_feature_augmentation(model, model.backbone, video, audio, targets, 4, register_forward_prehook_GAT)
        else:
            raise ValueError(f'Unknown net_type {net_type}.')
    
    def register_perturb_forward_prehook(self, model, net_type):
        if net_type == 'GAT':
            if isinstance(model, torch.nn.DataParallel):
                self.register_perturb_forward_prehook_layers(model.module, model.module.backbone, 4, register_forward_prehook_GAT)
            else:
                self.register_perturb_forward_prehook_layers(model, model.backbone, 4, register_forward_prehook_GAT)
        else:
            raise ValueError(f'Unknown net_type {net_type}.')

    def get_feature_augmentation(self, model, convnet, video, audio, targets, num_layers, register_func):
        layer_inputs = []
        def get_input_prehook(module, inp):
            inp[0].retain_grad()
            layer_inputs.append(inp[0])
        
        remove_handles = register_func(model, convnet, [get_input_prehook]*num_layers)
        video.requires_grad_()
        audio.requires_grad_()

        model.eval()
        # video_out, audio_out, logits = model(video, audio)
        logits = model(video, audio)['logits']
        # print('logits', logits)

        cls_loss = F.cross_entropy(logits, targets)
        model.zero_grad()
        cls_loss.backward()
        # print('cls_loss', cls_loss)

        # torch.save(layer_inputs, './debug_layer_inputs.tpy')
        inp_grads = [inp.grad.detach().clone() for inp in layer_inputs]
        # print(f'grad norms: {[grad.norm().item() for grad in inp_grads]}')
        video.requires_grad_(False)
        audio.requires_grad_(False)

        self.perturbations = inp_grads
        for p in self.perturbations:
            if torch.isnan(p).any():
                raise ValueError()
        
        for handle in remove_handles:
            handle.remove()
    
    def register_perturb_forward_prehook_layers(self, model, convnet, num_layers, register_func):
        def perturb_input_prehook_full(module: torch.nn.Module, inp, layer_id):
            # if len(self.perturbation_layers) > 0:
            #     print(f'perturb_0. {self.perturbation_layers}, {self.perturbation_idices_inbatch}, {self.perturbation_idices}, {self.perturbation_factor}')
            if layer_id in self.perturbation_layers:
                inp0 = inp[0].clone()
                p_layers = np.array(self.perturbation_layers)
                p_factor = np.array(self.perturbation_factor)
                p_idices = np.array(self.perturbation_idices)
                p_idices_inbatch = np.array(self.perturbation_idices_inbatch)
                # print('into perturb_0')
                p_index = np.nonzero(p_layers == layer_id)[0]
                num_new_axises = len(self.perturbations[layer_id].size()) - 1
                layer_perturbations = self.perturbations[layer_id].clone()
                layer_perturbations = layer_perturbations.to(inp0.device)
                if self.with_input_norm:
                    perturb = (inp0.data[p_idices_inbatch[p_index]].view(len(p_index), -1).norm(dim=-1) ** 2)[:, *(None,) * num_new_axises] * layer_perturbations[p_idices[p_index]] * torch.from_numpy(p_factor[p_index]).float()[:, *(None,) * num_new_axises].to(inp0.device)
                else:
                    perturb = self.perturbations[layer_id][p_idices[p_index]] * torch.from_numpy(p_factor[p_index]).float()[:, *(None,) * num_new_axises].to(inp0.device)
                # print(perturb.view(len(p_index), -1).norm(dim=-1))
                inp0[p_idices_inbatch[p_index]] += perturb
                return (inp0,)
        
        hooks = [partial(perturb_input_prehook_full, layer_id=i) for i in range(num_layers)]
        self.remove_handles.extend(register_func(model, convnet, hooks))

def register_forward_prehook_resnet32(model, convnet, hooks):
    remove_handles = []

    remove_handle_stage_1 = convnet.stage_1.register_forward_pre_hook(hooks[0])
    remove_handles.append(remove_handle_stage_1)
    remove_handle_stage_2 = convnet.stage_2.register_forward_pre_hook(hooks[1])
    remove_handles.append(remove_handle_stage_2)
    remove_handle_stage_3 = convnet.stage_3.register_forward_pre_hook(hooks[2])
    remove_handles.append(remove_handle_stage_3)
    remove_handle_fc = model.fc.register_forward_pre_hook(hooks[3])
    remove_handles.append(remove_handle_fc)

    return remove_handles

def register_forward_prehook_GAT(model, backbone, hooks):
    remove_handles = []

    remove_handle_stage_1 = backbone.GAT_layer1.register_forward_pre_hook(hooks[0])
    remove_handles.append(remove_handle_stage_1)
    remove_handle_stage_2 = backbone.GAT_layer2.register_forward_pre_hook(hooks[1])
    remove_handles.append(remove_handle_stage_2)
    remove_handle_stage_3 = backbone.GAT_layer3.register_forward_pre_hook(hooks[2])
    remove_handles.append(remove_handle_stage_3)
    remove_handle_fc = model.fc.register_forward_pre_hook(hooks[3])
    remove_handles.append(remove_handle_fc)

    return remove_handles
