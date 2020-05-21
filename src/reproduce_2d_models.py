#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import json
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


warnings.filterwarnings("ignore")


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
        "-d", "--dataset",
        dest="dataset",
        action="store", 
        help="set the dataset",
        choices=["Weissman", "Cpf1"],
        default="Weissman"
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
        default=99
    )
    parser.add_argument(
        "-u", "--use-mse",
        dest="mse",
        action="store_true", 
        help="use mse?",
        default=False
    )
    parser.add_argument(
        "-p", "--proportion",
        dest="proportion",
        action="store", 
        help="set proportion of the data (used for learning curve)",
        default="-1"
    )
    args = parser.parse_args()
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    torch.cuda.manual_seed(int(args.seed))
    with open(args.config, "r") as ih:
        config = json.load(ih)
    if args.dataset == "Weissman":
        data = pd.read_table(
            config["WeissmanDatasetPath"], index_col=0
        )
        series = data['perfect match sgRNA']
        print('series:', series.shape)
        val_series = np.random.choice(
            np.unique(series), size=int(len(np.unique(series))*.20), replace=False
        )
        val_indices = np.where(np.isin(series, val_series))
        train_indices = np.where(~np.isin(series, val_series))
        if args.proportion != "-1":
            train_indices, _ = train_test_split(train_indices, train_size=float(args.proportion))
        u = ImperfectMatchTransform("NGG", False, False, fold=False, cut_at_start=2, cut_at_end=1)
        transformer = transforms.Compose(
            [
                u, ToTensor(cudap=True)
            ]
        )
        train_set = WeissmanDataset(data, train_indices, transformer)
        val_set = WeissmanDataset(data, val_indices, transformer, n_bins=0)
        train_set_loader = DataLoader(train_set, shuffle=True, batch_size=config["batch_size"])
        val_set_loader = DataLoader(val_set, shuffle=True, batch_size=config["batch_size"])
        encoder = GuideHN2d(
            23, capsule_dimension=32, n_routes=1600, n_classes=5, n_channels=2,
        ).cuda()
        model = DKL(encoder, [1,5*32]).cuda().eval()
    EPOCHS = config["epochs"]
    print('X train:', len(train_set))
    print('X validation:', len(val_set))
    optimizer = Adam([
        {'params': model.parameters()}
    ], lr=0.01)
    mll = DeepApproximateMLL(VariationalELBOEmpirical(model.likelihood, model, config["batch_size"]))
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    training, validation = model.fit(
        train_set_loader, [val_set_loader], EPOCHS, 
        scheduler, optimizer, mll, args.output, rsquared, args.mse
    )
    y_hat = []
    y = []
    y_hat_std = []
    for i,b in tqdm(enumerate(val_set)):
        sequence, target = b
        y.append(float(target))
        ss = sequence.shape
        output, _ = model.forward(sequence.reshape(1, *ss))
        predictions = model.likelihood(output).mean.mean(0).cpu().data.numpy()
        y_hat_std.append(float(model.likelihood(output).variance.mean(0).cpu().data.numpy()[0]**0.5))
        y_hat.append(float(predictions[0]))
    with open(args.output+".json", "w") as oh:
        oh.write(
            json.dumps(
                {
                    "training": training, "validation": validation, "y": y, 
                    "y_hat": y_hat, "y_hat_std": y_hat_std
                }
            )
        )
