#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import umap
import json
import torch
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
from Bio import SeqIO
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
from scipy.stats import spearmanr, pearsonr
from catboost import CatBoostRegressor, Pool, cv
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error
from sklearn.metrics import accuracy_score, matthews_corrcoef, precision_score, recall_score
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
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


# In[4]:

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
        "-m", "--model",
        dest="model",
        action="store", 
        help="set the model file"
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
        "-n", "--n",
        dest="number",
        action="store", 
        help="n for top n guides",
        default=10
    )
    parser.add_argument(
        "-e", "--effector",
        dest="effector",
        action="store", 
        help="set the type of effector",
        choices=["Cas9", "Cpf1"],
        default="Cas9"
    )
    parser.add_argument(
        "-a", "--architecture",
        dest="architecture",
        action="store", 
        help="set the type of model",
        choices=["CNN", "RNN", "2D-CNN"],
        default="RNN"
    )
    args = parser.parse_args()
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    torch.cuda.manual_seed(int(args.seed))
    if not op.exists(args.output):
        os.makedirs(args.output)
        os.makedirs(op.join(args.output, "chr22_genes"))
        os.makedirs(op.join(args.output, "chr22_casoffinder_configs"))
        os.makedirs(op.join(args.output, "chr22_ordered"))
        os.makedirs(op.join(args.output, "chr22_grnas"))
    with open(args.config, "r") as ih:
        config = json.load(ih)
    ANNOTATION = config["Chr22_annotation"]
    annotation = pd.read_csv(ANNOTATION, header=1)[["Start", "Stop", "Gene symbol"]]
    annotation = list(
        zip(
            annotation["Start"].values, annotation["Stop"].values, 
            annotation["Gene symbol"].values
        )
    )
    annotation = sorted(annotation, key=lambda x: x[0])
    if args.effector == "Cas9":
        transformer = get_Cas9_transformer(args.architecture != "2D-CNN")
        TEMPLATE = "INTERMEDIATE/chr22_genes/GENE\nNNNNNNNNNNNNNNNNNNNNNGG\nNNNNNNNNNNNNNNNNNNNNNGG 0\n"
    elif args.effector == "Cpf1":
        transformer = get_Cpf1_transformer(args.architecture != "2D-CNN")
        TEMPLATE = "INTERMEDIATE/chr22_genes/GENE\nTTTNNNNNNNNNNNNNNNNNNNNN\nTTTNNNNNNNNNNNNNNNNNNNNN 0\n"
    chr22 = SeqIO.parse(config["Chr22_genome"], "fasta")
    chr22 = [a for a in chr22][0]
    COMMAND = "CASOFFINDER_PATH CONFIG G OUTPUT"
    TEMPLATE = TEMPLATE.replace("INTERMEDIATE", args.output)
    COMMAND = COMMAND.replace("CASOFFINDER_PATH", config["CASOFFINDERPATH"])
    if not op.exists(op.join(args.output, "run_casoffinder.sh")):
        script = open(op.join(args.output, "run_casoffinder.sh"), "w")
        script.write("#!/bin/sh\n\n")
        for start, stop, gene in tqdm(annotation):
            with open(args.output+"/chr22_genes/"+gene+".fa", "w") as oh:
                oh.write(">"+gene+"\n")
                oh.write(str(chr22.seq[start:stop]))
            with open(args.output+"/chr22_casoffinder_configs/"+gene+".txt", "w") as oh:
                oh.write(TEMPLATE.replace("GENE", gene+".fa"))
            script.write(
                COMMAND.replace(
                    "CONFIG", args.output+"/chr22_casoffinder_configs/"+gene+".txt"
                ).replace(
                    "OUTPUT", args.output+"/chr22_grnas/"+gene+".tsv"
                )+"\n"
            )
        script.close()
        os.system("sh "+op.join(args.output, "run_casoffinder.sh"))
    if args.architecture == "CNN":
        encoder = GuideHN(21, 32, 1360, n_classes=5).cuda()
        model = DKL(encoder, [1,5*32]).cuda()
        model.load_state_dict(torch.load(args.model))
        model = model.eval()
    elif args.architecture == "RNN":
        encoder = GuideHRNN(21, 32, 3360, n_classes=5).cuda()
        model = DKL(ncoder, [1,5*32]).cuda()
        model.load_state_dict(torch.load(args.model))
        model = model.eval()
    elif args.architecture == "2D-CNN":
        encoder = GuideHN2d(
            23, capsule_dimension=32, n_routes=1600, n_classes=5, n_channels=2,
        ).cuda()
        model = DKL(encoder, [1,5*32]).cuda().eval()
        model.load_state_dict(torch.load(args.model))
        model.eval()
    data = np.array(
        [a for a in os.walk(args.output+"/chr22_grnas/")][0][2]
    )
    sizes = np.array(
        [op.getsize(op.join(args.output+"/chr22_grnas", a)) for a in data]
    )
    genes = {a[2]: [] for a in annotation}
    for i,a in tqdm(enumerate(data)):
        print(i,a)
        current_df = pd.read_csv(
            op.join(
                args.output+"/chr22_grnas/"+a
            ), sep="\t", header=None
        ).dropna()
        current_df = current_df[current_df[3].apply(lambda x: len(x) == 23)]
        current_df[6] = [a]*current_df.shape[0]
        tds = DeepHFDataset(
            current_df, np.arange(current_df.shape[0]), transformer, sequence_column=3, 
            label_column=5
        )
        tld = DataLoader(tds, shuffle=False, batch_size=256)
        oa = []
        va = []
        for transformed_batch, _ in tqdm(tld):
            if args.architecture:
                transformed_batch = torch.stack([transformed_batch, transformed_batch])
                transformed_batch = transformed_batch.permute(1,0,2,3)
            tb = model(transformed_batch)
            o = model.likelihood(tb[0]).mean.mean(0).cpu().data.numpy()
            v = model.likelihood(tb[0]).variance.mean(0).cpu().data.numpy()
            oa.extend(o)
            va.extend(v)
        current_df[7] = oa
        current_df[8] = va
        current_df = current_df.sort_values(by=7, ascending=False)
        current_df.to_csv(
            args.output+"/chr22_ordered/"+a, sep="\t", header=None
        )
    ordered_data = [a for a in os.walk(args.output+"/chr22_ordered/")][0][2]
    total = None
    nothing = []
    for i,a in enumerate(ordered_data):
        print(i,a)
        try:
            current_df = pd.read_csv(
                args.output+"/chr22_ordered/"+a, sep="\t", header=None
            )
        except:
            nothing.append(a)
        else:
            if i == 0:
                total = current_df.head(int(args.number))
            else:
                total = total.append(current_df.head(int(args.number)))
    print("No gRNAs", nothing)
    with open(op.join(args.output, "nothing.txt"), "w") as oh:
        for a in nothing:
            oh.write(a+"\n")
    total.to_csv(
        op.join(
            args.output, args.model.replace("/", "_").replace(".", "").replace("ptch", "")+".tsv"
        ), sep="\t"
    )
