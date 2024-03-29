"""Script for training the DeepCRISPR model.

This script allows the user to train a DeepCRISPR model.

This script is not supposed to be ran as a module.
"""


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
from matplotlib.lines import Line2D
from torch.autograd import Variable
from matplotlib.patches import Patch
from torchvision import models, transforms
from torch.optim.lr_scheduler import StepLR
from scipy.stats import spearmanr, pearsonr
from gpytorch.priors import SmoothedBoxPrior
from torch.utils.data import DataLoader, Dataset
from gpytorch.models import ApproximateGP, ExactGP
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.variational import VariationalStrategy
from gpytorch.distributions import MultivariateNormal
from gpytorch.models.deep_gps import DeepGPLayer, DeepGP
from gpytorch.variational import CholeskyVariationalDistribution
from sklearn.model_selection import train_test_split, StratifiedKFold
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.kernels import GridInterpolationKernel
from gpytorch.mlls import VariationalELBO, VariationalELBOEmpirical
from gpytorch.mlls import DeepApproximateMLL
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import accuracy_score, matthews_corrcoef
from sklearn.metrics import precision_score, recall_score


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
        "-l", "--line",
        dest="line",
        action="store",
        help="set the cell line for test set",
        choices=["hela", "hek293t", "hl60", "hct116"],
        default="hela"
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
    parser.add_argument(
        "-v", "--validation",
        dest="validation",
        action="store_true",
        help="use leave-one-cell-line-out cross-validation?",
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
    if not op.exists(op.split(args.output)[0]):
        os.makedirs(op.split(args.output)[0])
    with open(args.config, "r") as ih:
        config = json.load(ih)
    deepCRISPRPATH = config["DeepCRISPRPath"]
    hct116 = pd.read_excel(
        deepCRISPRPATH, 0, engine="openpyxl"
    )
    hek293t = pd.read_excel(
        deepCRISPRPATH, 1, engine="openpyxl"
    )
    hela = pd.read_excel(
        deepCRISPRPATH, 2, engine="openpyxl"
    )
    hl60 = pd.read_excel(
        deepCRISPRPATH, 3, engine="openpyxl"
    )
    hl60_not_in_hct116 = np.logical_not(
        hl60["sgRNA"].isin(hct116["sgRNA"])
    ).values
    hl60_not_in_hek293t = np.logical_not(
        hl60["sgRNA"].isin(hek293t["sgRNA"])
    ).values
    hl60_not_in_hela = np.logical_not(
        hl60["sgRNA"].isin(hela["sgRNA"])
    ).values
    hela_not_in_hct116 = np.logical_not(
        hela["sgRNA"].isin(hct116["sgRNA"])
    ).values
    hela_not_in_hek293t = np.logical_not(
        hela["sgRNA"].isin(hek293t["sgRNA"])
    ).values
    hela_not_in_hl60 = np.logical_not(
        hela["sgRNA"].isin(hl60["sgRNA"])
    ).values
    hct116_not_in_hela = np.logical_not(
        hct116["sgRNA"].isin(hela["sgRNA"])
    ).values
    hct116_not_in_hek293t = np.logical_not(
        hct116["sgRNA"].isin(hek293t["sgRNA"])
    ).values
    hct116_not_in_hl60 = np.logical_not(
        hct116["sgRNA"].isin(hl60["sgRNA"])
    ).values
    hek293t_not_in_hela = np.logical_not(
        hek293t["sgRNA"].isin(hela["sgRNA"])
    ).values
    hek293t_not_in_hct116 = np.logical_not(
        hek293t["sgRNA"].isin(hct116["sgRNA"])
    ).values
    hek293t_not_in_hl60 = np.logical_not(
        hek293t["sgRNA"].isin(hl60["sgRNA"])
    ).values
    dfs = {
        "hct116": hct116[
            np.logical_and(
                hct116_not_in_hek293t, hct116_not_in_hl60, hct116_not_in_hela
            )
        ],
        "hela": hela[
            np.logical_and(
                hela_not_in_hek293t, hela_not_in_hl60, hela_not_in_hct116
            )
        ],
        "hl60": hl60[
            np.logical_and(
                hl60_not_in_hek293t, hl60_not_in_hct116, hl60_not_in_hela
            )
        ],
        "hek293t": hek293t[
            np.logical_and(
                hek293t_not_in_hct116, hek293t_not_in_hela,
                hek293t_not_in_hl60
            )
        ]
    }
    lines = list(dfs.keys())
    transformer = get_Cas9_transformer(True)
    if args.validation:
        current_test = dfs[args.line]
        lines.pop(lines.index(args.line))
        current_train = pd.concat([dfs[a] for a in lines])
        current_train_indices = np.arange(current_train.shape[0])
        if args.proportion != "-1":
            current_train_indices, _ = train_test_split(
                current_train_indices, train_size=float(args.proportion)
            )
        train_set = DeepHFDataset(
            current_train, current_train_indices, transform=transformer
        )
        val_set = DeepHFDataset(
            current_test, np.arange(current_test.shape[0]),
            transform=transformer
        )
    else:
        current = dfs[args.line]
        train_X, test_X, _, _ = train_test_split(
            np.arange(current.shape[0]),
            np.arange(current.shape[0]), test_size=0.2
        )
        if args.proportion != "-1":
            train_X, _ = train_test_split(
                train_X, train_size=float(args.proportion)
            )
        train_set = DeepHFDataset(
            current, train_X, transform=transformer
        )
        val_set = DeepHFDataset(
            current, test_X, transform=transformer
        )
    train_set_loader = DataLoader(train_set, shuffle=True, batch_size=256)
    val_set_loader = DataLoader(val_set, shuffle=True, batch_size=256)
    if args.model == "RNN":
        encoder = GuideHRNN(21, 32, 3360, n_classes=5).cuda()
    elif args.model == "CNN":
        encoder = GuideHN(21, 32, 1360, n_classes=5).cuda()
    model = DKL(encoder, [1, 5*32]).cuda().eval()
    EPOCHS = config["epochs"]
    if not op.exists(args.output):
        os.makedirs(args.output)
    print('X train:', len(train_set))
    print('X validation:', len(val_set))
    optimizer = Adam([
        {'params': model.parameters()}
    ], lr=0.01)
    mll = DeepApproximateMLL(
        VariationalELBOEmpirical(
            model.likelihood, model, config["batch_size"]
        )
    )
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    training, validation = model.fit(
        train_set_loader, [val_set_loader], EPOCHS,
        scheduler, optimizer, mll, args.output,
        lambda a, b: float(spearmanr(a, b)[0]), use_mse=args.mse
    )
    y_hat = []
    y = []
    y_hat_std = []
    for i, b in tqdm(enumerate(val_set)):
        sequence, target = b
        y.append(float(target))
        ss = sequence.shape
        output, _ = model.forward(sequence.reshape(1, *ss))
        predictions = model.likelihood(output).mean.mean(0).cpu().data.numpy()
        y_hat_std.append(
            float(
                model.likelihood(
                    output
                ).variance.mean(0).cpu().data.numpy()[0]**0.5
            )
        )
        y_hat.append(float(predictions[0]))
    with open(args.output+".json", "w") as oh:
        oh.write(
            json.dumps(
                {
                    "training": training, "validation": validation,
                    "y": y, "y_hat": y_hat, "y_hat_std": y_hat_std
                }
            )
        )
