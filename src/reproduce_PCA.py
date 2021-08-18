"""PCA training script

This script allows the user to train a PCA model to visualize the gRNA space.

This tool accepts comma separated value files (.csv) as well as excel
(.xls, .xlsx) files.

This script requires that `pandas` be installed within the Python
environment you are running this script in.

This file can also be imported as a module and contains the following
functions:

    * get_spreadsheet_cols - returns the column headers of the file
    * main - the main function of the script
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
from gpytorch.mlls import VariationalELBO
from gpytorch.mlls import VariationalELBOEmpirical, DeepApproximateMLL
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import accuracy_score, matthews_corrcoef
from sklearn.metrics import precision_score, recall_score
from sklearn.decomposition import PCA


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
        choices=["WT", "eSpCas9", "SpCas9HF1"],
        default="WT"
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
    args = parser.parse_args()
    if not op.exists(op.split(args.output)[0]):
        os.makedirs(op.split(args.output)[0])
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    torch.cuda.manual_seed(int(args.seed))
    with open(args.config, "r") as ih:
        config = json.load(ih)
    transformer = get_Cas9_transformer()
    DEEPHFPATH = config["DEEPHFPATH"]
    data = pd.read_excel(DEEPHFPATH, header=1, engine="openpyxl")
    if args.dataset == "WT":
        data = data[["21mer", "Wt_Efficiency"]].dropna()
        what = "Wt_Efficiency"
    elif args.dataset == "eSpCas9":
        data = data[["21mer", "eSpCas 9_Efficiency"]].dropna()
        what = "eSpCas 9_Efficiency"
    elif args.dataset == "SpCas9HF1":
        data = data[["21mer", "SpCas9-HF1_Efficiency"]].dropna()
        what = "SpCas9-HF1_Efficiency"
    train_X, testval_X, _, _ = train_test_split(
        np.arange(data.shape[0]), np.arange(data.shape[0]),
        test_size=0.085+0.15
    )
    print(train_X.shape)
    if args.proportion != "-1":
        train_X, _ = train_test_split(
            train_X, train_size=float(args.proportion)
        )
    print(train_X.shape)
    test_X, val_X, _, _ = train_test_split(
        testval_X, testval_X, test_size=0.085/(0.085+0.15)
    )
    train_set = DeepCRISPRDataset(
        data, np.arange(data.shape[0]), transformer, sequence_column="21mer", label_column=what
    )
    train_set_loader = DataLoader(train_set, shuffle=True, batch_size=256)
    if args.model == "RNN":
        encoder = GuideHRNN(21, 32, 3360, n_classes=5).cuda()
    elif args.model == "CNN":
        encoder = GuideHN(21, 32, 1360, n_classes=5).cuda()
    model = DKL(encoder, [1, 5*32]).cuda().eval()
    EPOCHS = config["epochs"]
    print('X train:', len(train_set))
    model.load_state_dict(torch.load(args.output+".ptch"))
    model = model.eval()
    tb_ra = []
    train_set_loader2 = DataLoader(train_set, shuffle=False, batch_size=256)
    for i, b in tqdm(enumerate(train_set_loader2)):
        sequences, targets = b
        tb_o = encoder(sequences)[0].cpu().data.numpy()
        tb_ra.extend(tb_o)
    tb_ra = np.stack(tb_ra).reshape(-1, 5*32)
    pvis_cm = PCA(random_state=1341)
    pv_cm = pvis_cm.fit_transform(tb_ra)
    with open(args.output+".pkl", "wb") as oh:
        pkl.dump(pvis_cm, oh)