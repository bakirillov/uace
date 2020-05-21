#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import json
import math
import torch
import warnings
import gpytorch
import argparse
import itertools
from core import *
import numpy as np
import pandas as pd
import pickle as pkl
from torch import nn
import os.path as op
from tqdm import tqdm
from time import time
from copy import deepcopy
from torch.optim import Adam
from tpot import TPOTRegressor
import torch.nn.functional as F
import matplotlib.pyplot as plt
from capsules.capsules import *
from IPython.display import Image
from qhoptim.pyt import QHM, QHAdam
from matplotlib.lines import Line2D
from torch.autograd import Variable
from matplotlib.patches import Patch
from torchvision import models, transforms
from torch.optim.lr_scheduler import StepLR
from scipy.stats import spearmanr, pearsonr
from gpytorch.priors import SmoothedBoxPrior
from torch.utils.data import DataLoader, Dataset
from catboost import CatBoostRegressor, Pool, cv
from gpytorch.models import ApproximateGP, ExactGP
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.variational import VariationalStrategy
from gpytorch.distributions import MultivariateNormal
from gpytorch.models.deep_gps import DeepGPLayer, DeepGP
from gpytorch.variational import CholeskyVariationalDistribution
from sklearn.model_selection import train_test_split, StratifiedKFold
from gpytorch.kernels import ScaleKernel, RBFKernel, GridInterpolationKernel
from gpytorch.mlls import VariationalELBO, VariationalELBOEmpirical, DeepApproximateMLL
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error
from sklearn.metrics import accuracy_score, matthews_corrcoef, precision_score, recall_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config",
        dest="config",
        action="store", 
        help="set the config file", 
        default="config.json"
    )
    parser.add_argument(
        "-o", "--output",
        dest="output",
        action="store", 
        help="set the path of output directory"
    )
    parser.add_argument(
        "-s", "--seed",
        dest="seed",
        action="store", 
        help="set the seed for prng",
        default=192
    )
    parser.add_argument(
        "-m", "--model",
        dest="model",
        action="store", 
        help="set the type of model",
        choices=["CNN", "RNN"],
        default="RNN"
    )
    parser.add_argument(
        "-u", "--use-mse",
        dest="mse",
        action="store_true", 
        help="use mse?",
        default=False
    )
    args = parser.parse_args()
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    torch.cuda.manual_seed(int(args.seed))
    with open(args.config, "r") as ih:
        config = json.load(ih)
    deepCpf1PATH = config["DeepCpf1KimPath"]
    train = pd.read_excel(op.join(deepCpf1PATH, "nbt.4061-S4.xlsx"), sheet_name=0, header=1)
    H1 = pd.read_excel(op.join(deepCpf1PATH, "nbt.4061-S4.xlsx"), sheet_name=1, header=1)
    H2 = pd.read_excel(op.join(deepCpf1PATH, "nbt.4061-S4.xlsx"), sheet_name=2, header=1)
    H3 = pd.read_excel(op.join(deepCpf1PATH, "nbt.4061-S4.xlsx"), sheet_name=3, header=1)
    train = train.iloc[np.arange(train.shape[0]-2)][:]
    H1 = H1.iloc[np.arange(H1.shape[0]-2)][:]
    H2 = H2.iloc[np.arange(H2.shape[0]-2)][:]
    H3 = H3.iloc[np.arange(H3.shape[0]-2)][:]
    train["label"] = train[train.columns[-1]].apply(lambda x: x/100 if x > 0 else 0)
    H1["label"] = H1[H1.columns[-1]].apply(lambda x: x/100 if x > 0 else 0)
    H2["label"] = H2[H2.columns[-1]].apply(lambda x: x/100 if x > 0 else 0)
    H3["label"] = H3[H3.columns[-1]].apply(lambda x: x/100 if x > 0 else 0)
    transformer = get_Cpf1_transformer()
    train_set = DeepHFDataset(
        train, np.arange(train.shape[0]), transform=transformer,
            sequence_column=train.columns[1], label_column="label"
        )
    H1_set = DeepHFDataset(
        H1, np.arange(H1.shape[0]), transform=transformer,
        sequence_column=H1.columns[1], label_column="label"
    )
    H2_set = DeepHFDataset(
        H2, np.arange(H2.shape[0]), transform=transformer,
        sequence_column=H2.columns[1], label_column="label"
    )
    H3_set = DeepHFDataset(
        H3, np.arange(H3.shape[0]), transform=transformer,
        sequence_column=H3.columns[1], label_column="label"
    )
    train_set_loader = DataLoader(train_set, shuffle=True, batch_size=256)
    H1_loader = DataLoader(H1_set, shuffle=True, batch_size=256)
    H2_loader = DataLoader(H2_set, shuffle=True, batch_size=256)
    H3_loader = DataLoader(H3_set, shuffle=True, batch_size=256)
    if args.model == "RNN":
        encoder = GuideHRNN(21, 32, 3360, n_classes=5).cuda()
    elif args.model == "CNN":
        encoder = GuideHN(21, 32, 1360, n_classes=5).cuda()
    model = DKL(encoder, [1,5*32]).cuda().eval()
    EPOCHS = config["epochs"]
    print('X train:', len(train_set))
    print('X H1:', len(H1_set))
    print('X H2:', len(H2_set))
    print('X H3:', len(H3_set))
    optimizer = Adam([
        {'params': model.parameters()}
    ], lr=0.01)
    mll = DeepApproximateMLL(VariationalELBOEmpirical(model.likelihood, model, config["batch_size"]))
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    training, validation = model.fit(
        train_set_loader, [H1_loader, H2_loader, H3_loader], EPOCHS, 
        scheduler, optimizer, mll, args.output, lambda a,b: float(spearmanr(a, b)[0]), use_mse=args.mse
    )
    y_hat_H1 = []
    y_H1 = []
    y_hat_std_H1 = []
    for i,b in tqdm(enumerate(H1_set)):
        sequence, target = b
        y_H1.append(float(target))
        ss = sequence.shape
        output, _ = model.forward(sequence.reshape(1, *ss))
        predictions = model.likelihood(output).mean.mean(0).cpu().data.numpy()
        y_hat_std_H1.append(float(model.likelihood(output).variance.mean(0).cpu().data.numpy()[0]**0.5))
        y_hat_H1.append(float(predictions[0]))
    y_hat_H2 = []
    y_H2 = []
    y_hat_std_H2 = []
    for i,b in tqdm(enumerate(H2_set)):
        sequence, target = b
        y_H2.append(float(target))
        ss = sequence.shape
        output, _ = model.forward(sequence.reshape(1, *ss))
        predictions = model.likelihood(output).mean.mean(0).cpu().data.numpy()
        y_hat_std_H2.append(float(model.likelihood(output).variance.mean(0).cpu().data.numpy()[0]**0.5))
        y_hat_H2.append(float(predictions[0]))
    y_hat_H3 = []
    y_H3 = []
    y_hat_std_H3 = []
    for i,b in tqdm(enumerate(H3_set)):
        sequence, target = b
        y_H3.append(float(target))
        ss = sequence.shape
        output, _ = model.forward(sequence.reshape(1, *ss))
        predictions = model.likelihood(output).mean.mean(0).cpu().data.numpy()
        y_hat_std_H3.append(float(model.likelihood(output).variance.mean(0).cpu().data.numpy()[0]**0.5))
        y_hat_H3.append(float(predictions[0]))
    with open(args.output+".json", "w") as oh:
        oh.write(
            json.dumps(
                {
                    "training": training, "validation": validation, 
                    "y_H1": y_H1, "y_hat_H1": y_hat_H1, "y_hat_std_H1": y_hat_std_H1,
                    "y_H2": y_H2, "y_hat_H2": y_hat_H2, "y_hat_std_H2": y_hat_std_H2,
                    "y_H3": y_H3, "y_hat_H3": y_hat_H3, "y_hat_std_H3": y_hat_std_H3
                }
            )
        )