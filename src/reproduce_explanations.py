#!/usr/bin/env python
# coding: utf-8

# In[149]:


import os
import umap
import json
import torch
import itertools
from core import *
import numpy as np
import pandas as pd
import pickle as pkl
from torch import nn
import os.path as op
from tqdm import tqdm
from time import time
from weblogolib import *
from copy import deepcopy
from torch.optim import Adam
from tpot import TPOTRegressor
import torch.nn.functional as F
import matplotlib.pyplot as plt
from capsules.capsules import *
from matplotlib import rcParams
from IPython.display import Image
from matplotlib.lines import Line2D
from torch.autograd import Variable
from matplotlib.patches import Patch
from torch.optim.lr_scheduler import StepLR
from scipy.stats import spearmanr, pearsonr
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error
from sklearn.metrics import accuracy_score, matthews_corrcoef, precision_score, recall_score


# In[2]:


from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset


# In[3]:


import gpytorch
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.models import ApproximateGP, ExactGP 
from sklearn.metrics import accuracy_score
from gpytorch.priors import SmoothedBoxPrior
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.kernels import ScaleKernel, RBFKernel, GridInterpolationKernel
from gpytorch.mlls import VariationalELBO, VariationalELBOEmpirical, DeepApproximateMLL
from gpytorch.distributions import MultivariateNormal
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
from gpytorch.models.deep_gps import DeepGPLayer, DeepGP

from scipy.special import softmax
# In[24]:


from alibi.explainers import ALE
import argparse


class ALEDataset(Dataset):

    def __init__(self, x):
        self.x = x
        
    def __len__(self):
        return(self.x.shape[0])

    def __getitem__(self, ind):
        return(self.x[ind])


class ModelForALE():
    
    def __init__(self, encoder, model, shape, mean=True):
        self.M = model
        self.E = encoder
        self.shape = shape
        
    def predict_single(self, x):
        X_h = x.reshape(-1, *self.shape).to(0)
        tb = self.M(X_h)
        l = self.M.likelihood(tb[0])
        out_m = l.mean.mean(0).cpu().data.numpy()
        out_v = l.variance.mean(0).cpu().data.numpy()
        out = np.stack([out_m, out_v])
        return(out.T)
    
    def predict(self, x):
        ds = ALEDataset(x)
        ld = DataLoader(ds, shuffle=False, batch_size=256)
        out = []
        for a in tqdm(ld):
            out.append(self.predict_single(a))
        return(np.concatenate(out))

    
def make_logo(ale_vals, input_size, filename, temperature=0.01, what="mean efficiency"):
    ale_vals = softmax(ale_vals/temperature, 0).T
    ld = LogoData.from_counts("AGCT", ale_vals)
    logooptions = LogoOptions()
    logooptions.color_scheme = classic
    logooptions.title = "ALE Logo for "+what
    logoformat = LogoFormat(ld, logooptions)
    png = png_formatter(ld, logoformat)
    with open(filename, "wb") as oh:
        oh.write(png)
        
def get_m(rownames, colnames, INPUT_SIZE, seqnames=None):
    if seqnames:
        m_shape = (2, 4, INPUT_SIZE)
        m = np.zeros(shape=m_shape).astype("str")
        for k,c in enumerate(seqnames):
            for i,a in enumerate(rownames):
                for j,b in enumerate(colnames):
                    m[k,i,j] = a+"_at_"+b+"_in_"+c
    else: 
        m_shape = (4, INPUT_SIZE)
        m = np.zeros(shape=m_shape).astype("str")
        for i,a in enumerate(rownames):
            for j,b in enumerate(colnames):
                m[i,j] = a+"_at_"+b
    return(m.reshape(np.product(m_shape)).tolist(), m_shape)
    
def make_random(n, INPUT_SIZE, transformer, two=False):
    random_set = []
    if two:
        for a in range(n):
            random_set.append(
                transformer(
                    "".join(
                        np.random.choice(["A", "T", "G", "C"], INPUT_SIZE)
                    )+","+"".join(np.random.choice(["A", "T", "G", "C"], INPUT_SIZE))
                ).cpu().data.numpy()
            )
    else:
        for a in range(n):
            random_set.append(
                transformer("".join(np.random.choice(["A", "T", "G", "C"], INPUT_SIZE))).cpu().data.numpy()
            )
    random_set = np.stack(random_set)
    return(random_set)
   
def init_names(INPUT_SIZE, ENCODER):
    rownames = ["A", "G", "C", "T"]
    colnames = [str(a) for a in range(1, INPUT_SIZE+1)]
    seqnames = ["gRNA", "target"] if ENCODER == "2D-CNN" else ["gRNA"]
    return(rownames, colnames, seqnames)
    
    
              
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model",
        dest="model",
        action="store",
        help="the filename of the model file"
    )
    parser.add_argument(
        "-a", "--architecture",
        dest="architecture",
        action="store",
        help="the architecture description",
        default="CNN.21",
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
        "-e", "--effector",
        dest="effector",
        action="store",
        help="set the effector",
        choices=["Cas9", "Cpf1"],
        default="Cas9"
    )
    parser.add_argument(
        "-n", "--number",
        dest="number",
        action="store",
        help="set the number of random sequences to compute the explanations for",
        default="10000"
    )
    parser.add_argument(
        "-t", "--temperature",
        dest="temperature",
        action="store",
        help="set the temperature for softmax",
        default="0.01,0.0001"
    )
    args = parser.parse_args()
    if not op.exists(args.output):
        os.makedirs(args.output)
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    torch.cuda.manual_seed(int(args.seed))
    ENCODER = args.architecture.split(".")[0]
    INPUT_SIZE = int(args.architecture.split(".")[1])
    CAPS = {
        "CNN.21": 1360, "RNN.21": 3360, "2D-CNN.23": 1600
    }
    if ENCODER in ["CNN", "RNN"]:
        if args.effector == "Cas9":
            transformer = get_Cas9_transformer(False)
        else:
            transformer = get_Cpf1_transformer(False, cut_at_start=0, cut_at_end=0)
    else:
        if args.effector == "Cas9":
            u = ImperfectMatchTransform("NGG", False, False, fold=False, cut_at_start=0, cut_at_end=0)
            transformer = transforms.Compose(
                [
                    u, ToTensor(cudap=True)
                ]
            )
        else:
            u = ImperfectMatchTransform(
                "TTTN", False, False, fold=False, cut_at_start=0, cut_at_end=0
            )
            transformer = transforms.Compose(
                [
                    u, ToTensor(cudap=True)
                ]
            )
    if ENCODER == "CNN":
        encoder = GuideHN(INPUT_SIZE, 32, CAPS[args.architecture], n_classes=5).cuda()
    elif ENCODER == "RNN":
        encoder = GuideHRNN(INPUT_SIZE, 32, CAPS[args.architecture], n_classes=5).cuda()
    elif ENCODER == "2D-CNN":
        encoder = GuideHN2d(
            INPUT_SIZE, capsule_dimension=32, n_routes=CAPS[args.architecture], n_classes=5, n_channels=2,
        ).cuda()
    model = DKL(encoder, [1,5*32]).cuda()
    model.load_state_dict(torch.load(args.model))
    model = model.eval()
    rownames, colnames, seqnames = init_names(INPUT_SIZE, ENCODER)
    #m_shape = (2, 4, INPUT_SIZE) if ENCODER == "2D-CNN" else (4, INPUT_SIZE)
    #m = np.zeros(shape=m_shape).astype("str")
    if ENCODER == "2D-CNN":
        m, m_shape = get_m(rownames, colnames, INPUT_SIZE, seqnames=seqnames)
    else:
        m, m_shape = get_m(rownames, colnames, INPUT_SIZE)
    #m = m.reshape(np.product(m_shape)).tolist()
    mfa = ModelForALE(encoder, model, m_shape)
    ale = ALE(mfa.predict, m, ["mean", "variance"])
    random_set = make_random(int(args.number), INPUT_SIZE, transformer, ENCODER == "2D-CNN")
    random_set = random_set.reshape(-1, np.product(m_shape))
    explanation = ale.explain(random_set)
    ale_values = np.stack(explanation.ale_values).squeeze()
    figname = op.join(
        args.output, args.effector+"."+args.architecture+op.split(args.model)[-1]+".png"
    )
    if ENCODER == "2D-CNN":
        rcParams["figure.figsize"] = (10, 6)
        fig,ax = plt.subplots(2,2)
        ax[0][0].set_title("Mean Efficiency ~ Position+Nucleotide+gRNA")
        ax[0][0].imshow(ale_values.T[0][1][0:92].reshape(4, INPUT_SIZE))
        ax[0][0].set_yticks([0,1,2,3])
        ax[0][0].set_yticklabels(["A", "G", "C", "T"])
        ax[1][0].set_title("Mean Efficiency ~ Position+Nucleotide+target")
        ax[1][0].imshow(ale_values.T[0][1][92:].reshape(4, INPUT_SIZE))
        ax[1][0].set_yticks([0,1,2,3])
        ax[1][0].set_yticklabels(["A", "G", "C", "T"])
        ax[0][1].set_title("Variance ~ Position+Nucleotide+gRNA")
        ax[0][1].imshow(ale_values.T[1][1][0:92].reshape(4, INPUT_SIZE))
        ax[0][1].set_yticks([0,1,2,3])
        ax[0][1].set_yticklabels(["A", "G", "C", "T"])
        ax[1][1].set_title("Variance ~ Position+Nucleotide+target")
        ax[1][1].imshow(ale_values.T[1][1][92:].reshape(4, INPUT_SIZE))
        ax[1][1].set_yticks([0,1,2,3])
        ax[1][1].set_yticklabels(["A", "G", "C", "T"])
        make_logo(
            ale_values.T[0][1][0:92].reshape(4, INPUT_SIZE), INPUT_SIZE, 
            figname.replace(".png", ".gRNA.mean.logo.png"), 
            temperature=float(args.temperature.split(",")[0]), what="mean efficiency"
        )
        make_logo(
            ale_values.T[1][1][0:92].reshape(4, INPUT_SIZE), INPUT_SIZE, 
            figname.replace(".png", ".gRNA.var.logo.png"), 
            temperature=float(args.temperature.split(",")[1]), what="variance"
        )
        make_logo(
            ale_values.T[0][1][92:].reshape(4, INPUT_SIZE), INPUT_SIZE, 
            figname.replace(".png", ".target.mean.logo.png"), 
            temperature=float(args.temperature.split(",")[2]), what="mean efficiency"
        )
        make_logo(
            ale_values.T[1][1][92:].reshape(4, INPUT_SIZE), INPUT_SIZE, 
            figname.replace(".png", ".target.var.logo.png"), 
            temperature=float(args.temperature.split(",")[3]), what="variance"
        )
    else:
        make_logo(
            ale_values.T[0][1].reshape(4, INPUT_SIZE), INPUT_SIZE, 
            figname.replace(".png", ".mean.logo.png"), 
            temperature=float(args.temperature.split(",")[0]), what="mean efficiency"
        )
        make_logo(
            ale_values.T[1][1].reshape(4, INPUT_SIZE), INPUT_SIZE, 
            figname.replace(".png", ".var.logo.png"), 
            temperature=float(args.temperature.split(",")[1]), what="variance"
        )
        rcParams["figure.figsize"] = (10, 3)
        fig,ax = plt.subplots(1,2)
        ax[0].set_title("Mean Efficiency ~ Position+Nucleotide")
        ax[0].imshow(ale_values.T[0][1].reshape(4, INPUT_SIZE))
        ax[0].set_yticks([0,1,2,3])
        ax[0].set_yticklabels(["A", "G", "C", "T"])
        ax[1].set_title("Variance ~ Position+Nucleotide")
        ax[1].imshow(ale_values.T[1][1].reshape(4, INPUT_SIZE))
        ax[1].set_yticks([0,1,2,3])
        ax[1].set_yticklabels(["A", "G", "C", "T"])
    fig.savefig(figname)
    fig.show()