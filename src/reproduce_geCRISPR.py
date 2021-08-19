"""Script for training the geCRISPR model.

This script allows the user to train a geCRISPR model.

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
from gpytorch.kernels import ScaleKernel, RBFKernel, GridInterpolationKernel
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
        "-p", "--proportion",
        dest="proportion",
        action="store",
        help="set proportion of the data (used for learning curve)",
        default="-1"
    )
    parser.add_argument(
        "-f", "--folds",
        dest="folds",
        action="store",
        help="set the folds file",
        default=None
    )
    parser.add_argument(
        "-n", "--number",
        dest="fold_number",
        action="store",
        help="set the folds number",
        default=0
    )
    args = parser.parse_args()
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    torch.cuda.manual_seed(int(args.seed))
    if not op.exists(op.split(args.output)[0]):
        os.makedirs(op.split(args.output)[0])
    with open(args.config, "r") as ih:
        config = json.load(ih)
    transformer = get_Cas9_transformer()
    T3619PATH = config["T3619PATH"]
    V520PATH = config["V520PATH"]
    if not op.exists(op.split(args.output)[0]):
        os.makedirs(op.split(args.output)[0])
    print(args.folds)
    if args.folds == None:
        train_X_T3619, test_X_T3619, _, _ = train_test_split(
            np.arange(3619), np.arange(3619), test_size=0.2
        )
        print(train_X_T3619)
        if args.proportion != "-1":
            train_X_T3619 = train_test_split(
                train_X_T3619, train_size=float(args.proportion)
            )[0]
            print(train_X_T3619)
        train_set = GeCRISPRDataset(
            T3619PATH, train_X_T3619, transform=transformer, classification=False
        )
        val_set = GeCRISPRDataset(
            T3619PATH, test_X_T3619, transform=transformer, classification=False
        )
        test_set = GeCRISPRDataset(
            V520PATH, np.arange(520), transform=transformer, classification=False
        )
    else:
        with open(args.folds, "rb") as ih:
            kf = pkl.load(ih)
        train, test = kf[int(args.fold_number)]
        train_set = GeCRISPRDataset(
            T3619PATH, train, transform=transformer, classification=False
        )
        val_set = GeCRISPRDataset(
            T3619PATH, test, transform=transformer, classification=False
        )
        test_set = GeCRISPRDataset(
            T3619PATH, test, transform=transformer, classification=False
        )
    train_set_loader = DataLoader(train_set, shuffle=True, batch_size=256)
    val_set_loader = DataLoader(val_set, shuffle=True, batch_size=256)
    if args.model == "RNN":
        encoder = GuideHRNN(20, 32, 3200, n_classes=5)#.cuda()
    elif args.model == "CNN":
        encoder = GuideHN(20, 32, 1280, n_classes=5)#.cuda()
    model = DKL(encoder, [1, 5*32]).eval()#.cuda().eval()
    EPOCHS = config["epochs"]
    print('X train:', len(train_set))
    print('X validation:', len(val_set))
    optimizer = Adam([
        {'params': model.parameters()}
    ], lr=0.01)
    mll = DeepApproximateMLL(
        VariationalELBOEmpirical(model.likelihood, model, config["batch_size"])
    )
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    training, validation = model.fit(
        train_set_loader, [val_set_loader], EPOCHS,
        scheduler, optimizer, mll, args.output,
        lambda a, b: float(pearsonr(a, b)[0]), use_mse=args.mse, use_cuda=False
    )
    y_hat_T = []
    y_T = []
    y_hat_std_T = []
    train_set_loader2 = DataLoader(train_set, shuffle=False, batch_size=256)
    for i, b in tqdm(enumerate(train_set_loader2)):
        sequences, targets = b
        y_T.extend([float(a) for a in targets.cpu().data.numpy()])
        output, _ = model.forward(sequences)
        predictions = model.likelihood(output).mean.mean(0).cpu().data.numpy()
        y_hat_std_T.extend(
            [
                float(a) for a in model.likelihood(
                    output
                ).variance.mean(0).cpu().data.numpy()**0.5
            ]
        )
        y_hat_T.extend(
            [float(a) for a in predictions]
        )
    y_hat = []
    y = []
    y_hat_std = []
    test_set_loader2 = DataLoader(test_set, shuffle=False, batch_size=256)
    for i, b in tqdm(enumerate(test_set_loader2)):
        sequences, targets = b
        y.extend([float(a) for a in targets.cpu().data.numpy()])
        output, _ = model.forward(sequences)
        predictions = model.likelihood(output).mean.mean(0).cpu().data.numpy()
        y_hat_std.extend(
            [
                float(a) for a in model.likelihood(
                    output
                ).variance.mean(0).cpu().data.numpy()**0.5
            ]
        )
        y_hat.extend(
            [float(a) for a in predictions]
        )
    jsonfile = args.output+".json" if args.folds == None else args.output+"."+str(args.fold_number)+".json"
    with open(jsonfile, "w") as oh:
        oh.write(
            json.dumps(
                {
                    "training": training, "validation": validation,
                    "y": y, "y_hat": y_hat, "y_hat_std": y_hat_std,
                    "y_T": y_T, "y_hat_T": y_hat_T, "y_hat_std_T": y_hat_std_T
                }
            )
        )
