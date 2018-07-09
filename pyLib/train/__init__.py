import torch
import scipy.linalg
from .train import trainer, getFileName, genFromDefaultConfig, trainOne, trainAutoEncoder
from .weighttrain import weightTrainer
from .dataLoader import dataLoader, keyFactory, vecKeyFactory, labelFactory, subFactory, Factory, unaryKeyFactory
from .torchUtil import GaoNet, modelLoader, modelLoaderV2, plotError, encoderLoader, svcLoader, recordStep0, model2cpu
from .torchUtil import autoEncoder as AutoEncoder
from .MoM import MoMNet, momLoader
from ..math.stat import getStandardData

# use pretty names
from .train import trainer as Trainer
from .train import genFromDefaultConfig as genTrainConfig
from .dataLoader import dataLoader as DataLoader
from .dataLoader import keyFactory as KeyFactory
from .dataLoader import vecKeyFactory as VecKeyFactory
from .dataLoader import labelFactory as LabelFactory
from .dataLoader import subFactory as SubFactory
from .dataLoader import unaryKeyFactory as UnaryKeyFactory
