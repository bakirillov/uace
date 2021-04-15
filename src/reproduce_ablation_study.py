#!/usr/bin/env python
# coding: utf-8

# In[2]:


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


# In[3]:


from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset


# In[4]:


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


# In[6]:


from collections import Counter
from keras.layers import Conv2D, Activation, MaxPool2D, Flatten, Dense, Dropout
from keras.models import Sequential


# In[7]:


np.random.seed(99)
torch.manual_seed(99)
torch.cuda.manual_seed(99)


def binarize_sequence(sequence, length=23):
    """
    converts a 26-base nucleotide string to a binarized array of shape (4,26)
    """
    arr = np.zeros((4, length))
    for i in range(length):
        if sequence[i] == 'A':
            arr[0,i] = 1
        elif sequence[i] == 'C':
            arr[1,i] = 1
        elif sequence[i] == 'G':
            arr[2,i] = 1
        elif sequence[i] == 'T':
            arr[3,i] = 1
        else:
            raise(
                Exception(
                    'sequence contains characters other than A,G,C,T \n%s'%sequence
                )
            )
    return(arr)


# In[9]:

def train_JostSet_for_original(start=2, end=-1, length=23):
    data = pd.read_csv(
        "/home/bakirillov/Data/Svetlana_data/Table_S8_machine_learning_input.txt", index_col=0, sep="\t"
    )
    sequence_tuples = list(zip(data['genome input'], data['sgRNA input']))
    test_sequence = sequence_tuples[0][0][start:end]
    print(start, end, length)
    print(test_sequence)
    print(binarize_sequence(test_sequence, length))
    print(binarize_sequence(test_sequence, length).shape)
    stacked_arrs = [
        np.stack(
            (binarize_sequence(genome_input[start:end], length), binarize_sequence(sgrna_input[start:end], length)), axis=2
        ) 
            for (genome_input, sgrna_input) in sequence_tuples]
    X = np.concatenate([arr[np.newaxis] for arr in stacked_arrs])
    y = data['mean relative gamma'].values
    series = data['perfect match sgRNA']
    print('X:', X.shape)
    print('y:', y.shape)
    print('series:', series.shape)
    val_series = np.random.choice(
        np.unique(series), size=int(len(np.unique(series))*.20), replace=False
    )
    val_indices = np.where(np.isin(series, val_series))
    train_indices = np.where(~np.isin(series, val_series))
    X_train = X[train_indices]
    X_val = X[val_indices]
    y_train = y[train_indices]
    y_val = y[val_indices]
    nbins=5
    y_train_clipped = y_train.clip(0,1)
    y_train_binned, histbins = pd.cut(
        y_train_clipped, np.linspace(0,1,nbins+1), 
        labels=range(nbins), include_lowest=True, retbins=True
    )
    print('bin edges:', histbins)
    class_weights = {k:1/float(v) for k,v in Counter(y_train_binned).items()}
    class_weights[0] = class_weights[0] * 1.5
    class_weights = {k:v/sum(class_weights.values()) for k,v in class_weights.items()}
    sample_weights = [class_weights[Y] for Y in y_train_binned]
    print(class_weights)
    model = Sequential()
    model.add(
        Conv2D(
            filters=32, kernel_size=(4,4), activation='relu', 
            padding='same', input_shape=(4,length,2), data_format='channels_last'
        )
    )
    model.add(Conv2D(filters=32, kernel_size=(4,4), activation='relu', padding='same', data_format='channels_last'))
    model.add(MaxPool2D(pool_size=(1,2), padding='same', data_format='channels_last'))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(units=128, activation='sigmoid'))
    model.add(Dropout(0.1))
    model.add(Dense(units=32, activation='sigmoid'))
    model.add(Dropout(0.1))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='logcosh', metrics=['mse'], optimizer='adam')
    model_history = model.fit(X_train, 
                          y_train.ravel(), 
                          sample_weight=np.array(sample_weights), 
                          batch_size=32, 
                          epochs=8, 
                          validation_data=(X_val, y_val.ravel()))
    print('r squared = %.3f'%pearsonr(y_val, model.predict(X_val).ravel())[0]**2)
    return(model, pearsonr(y_val, model.predict(X_val).ravel())[0]**2)
