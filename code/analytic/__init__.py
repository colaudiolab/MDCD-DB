# -*- coding: utf-8 -*-
from .Learner import Learner
from .Buffer import Buffer, RandomBuffer, GaussianKernel
from .AnalyticLinear import AnalyticLinear, RecursiveLinear
from .ACIL import ACIL, ACILLearner
from .DSAL import DSAL, DSALLearner
from .Finetune import FinetuneLearner
from .PASS import PassLearner
from .iCaRL import iCaRLLearner
from .LWF import LWFLearner
from .Fetril import FetrilLearner
from .MMAL import MMALLearner
from .Replay import ReplayLearner
from .DFIL import DFILLearner
from .MyTagFex import TagFexLearner

__all__ = [
    "Learner",
    "Buffer",
    "RandomBuffer",
    "GaussianKernel",
    "AnalyticLinear",
    "RecursiveLinear",
    "ACIL",
    "DSAL",
    "ACILLearner",
    "DSALLearner",
    "FinetuneLearner",
    "PassLearner",
    "iCaRLLearner",
    "LWFLearner",
    "FetrilLearner"
    "MMALLearner",
    "ReplayLearner",
    "DFILLearner",
    "TagFexLearner",
]
