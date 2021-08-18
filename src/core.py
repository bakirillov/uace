"""The module with all required definitions

This file provides the definitions to build every model.

This file is to be imported as a module and exports the following:

    get_Cpf1_transformer - Construct a 
        transforms.Compose to preprocess Cas12a gRNA data
    get_Cas9_transformer -  Construct a 
        transforms.Compose to preprocess Cas9 gRNA data
    in_CI - Computes p_68, p_95 and p_99.7
    rsquared - Computes R^2 between a and b
    onehot - Performs a one-hot encoding 
        of nucleotide sequence
    correct_order - Reshapes the one-hot 
        encoding array u to (4, length of string)
    CoordPrimaryCapsuleLayer - A class to define 
        primary capsules with CoordConv
    OneHotAndCut - Builds CoordConv1d kernels 
        according to the specified parameters
    ToTensor - A helper class to make 
        a tensor out ouf preprocessing
    PengDataset - A torch Dataset for Peng et al. dataset
    DeepCRISPRDataset - A torch Dataset for 
        DeepCRISPR dataset and similar ones
    plot_LC - Plot learning curves
    FastaSet - A torch Dataset for 
        single record fasta files
    CasOffFinderSet - A torch Dataset for 
        CasOffFinder output
    GuideHN - Guide Hit-or-Miss network with CNN
    GuideHRNN - Guide Hit-or-Miss network with RNN
    GaussianProcessLayer - Gaussian Process Layer
    DKL - Main Deep Kernel Learning module
    JostEtAlDataset - A torch Dataset for Jost et al. data
    ImperfectMatchTransform - Defines the preprocessing 
        for gRNA-target pairs
    GuideHN2d - Guide Hit-or-Miss network with 2D-CNN
    DeepHFDataset - A torch Dataset for DeepHF data
    GeCRISPRDataset - A torch Dataset for geCRISPR data
"""

import re
import os
import math
import torch
import joblib
import gpytorch
import itertools
import numpy as np
import pandas as pd
from torch import nn
import os.path as op
from tqdm import tqdm
from time import time
from copy import deepcopy
from torch.optim import Adam
from collections import Counter
import torch.nn.functional as F
import matplotlib.pyplot as plt
from capsules.capsules import *
from CoordConv.coordconv import *
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.metrics import accuracy_score
from torchvision import models, transforms
from scipy.stats import pearsonr, spearmanr
from gpytorch.priors import SmoothedBoxPrior, NormalPrior
from torch.utils.data import DataLoader, Dataset
from gpytorch.models import ApproximateGP, ExactGP
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.variational import VariationalStrategy
from gpytorch.distributions import MultivariateNormal
from Bio.SeqFeature import SeqFeature, FeatureLocation
from gpytorch.models.deep_gps import DeepGPLayer, DeepGP
from gpytorch.variational import CholeskyVariationalDistribution
from sklearn.model_selection import train_test_split, StratifiedKFold
from gpytorch.kernels import ScaleKernel, RBFKernel, GridInterpolationKernel, Kernel
from gpytorch.mlls import VariationalELBO, VariationalELBOEmpirical, GammaRobustVariationalELBO
from gpytorch.mlls import DeepApproximateMLL
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import median_absolute_error
from math import exp, pi
from gpytorch import constraints


def get_Cpf1_transformer(cut_pam=True, cut_at_start=4, cut_at_end=6):
    """
    Construct a transforms.Compose to preprocess Cas12a gRNA data

    cut_pam flag is set if the PAM sequence is supposed to be cut
    cut_at_start specifies how much we need to cut the flanks 
                at the start of the sequence
    cut_at_end specifies how much we need to cut the flanks 
                at the end of the sequence

    Outputs a transform.Compose instance
    """
    u = OneHotAndCut(
        "TTTN", True, cut_pam, fold=False, 
        cut_at_start=cut_at_start, cut_at_end=cut_at_end
    )
    transformer = transforms.Compose(
        [
            u, ToTensor(cudap=True)
        ]
    )
    return(transformer)

def get_Cas9_transformer(cut_pam=False):
    """
    Construct a transforms.Compose to preprocess Cas9 gRNA data

    cut_pam flag is set if the PAM sequence is supposed to be cut

    Outputs a transform.Compose instance
    """
    u = OneHotAndCut("NGG", False, cut_pam, fold=False)
    transformer = transforms.Compose(
        [
            u, ToTensor(cudap=True)
        ]
    )
    return(transformer)

def in_CI(ys, y_hats, stds):
    """
    Computes p_68, p_95 and p_99.7
    
    ys specifies the real labels
    y_hats specifies predictive means
    std specifies standard deviations of predictions
    
    Outputs a pd.DataFrame with the values
    """
    confs = {
        "0.68": [],
        "0.95": [],
        "0.997": []
    }
    for y,y_hat,std in zip(ys, y_hats, stds):
        one_sigma = [y_hat-std, y_hat+std]
        if y>=one_sigma[0] and y<=one_sigma[1]:
            confs["0.68"].append(1)
        else:
            confs["0.68"].append(0)
        two_sigma = [y_hat-2*std, y_hat+2*std]
        if y>=two_sigma[0] and y<=two_sigma[1]:
            confs["0.95"].append(1)
        else:
            confs["0.95"].append(0)
        three_sigma = [y_hat-3*std, y_hat+3*std]
        if y>=three_sigma[0] and y<=three_sigma[1]:
            confs["0.997"].append(1)
        else:
            confs["0.997"].append(0)
    return(pd.DataFrame(confs))

def rsquared(a, b):
    """Computes R^2 between a and b"""
    return(float(pearsonr(a, b)[0]**2))

def onehot(u):
    """
    Performs a one-hot encoding of nucleotide sequence
    
    u is the input string
    
    Outputs a 1D numpy array with the encoding 
    """
    encoding = {
        1: [1,0,0,0],
        2: [0,0,0,1],
        3: [0,1,0,0],
        4: [0,0,1,0],
        0: [0,0,0,0],
        "A": [1,0,0,0],
        "T": [0,0,0,1],
        "G": [0,1,0,0],
        "C": [0,0,1,0],
        "N": [0,0,0,0],
        "a": [1,0,0,0],
        "t": [0,0,0,1],
        "g": [0,1,0,0],
        "c": [0,0,1,0],
        "n": [0,0,0,0],
        "(": [0,0,1],
        ")": [0,1,0],
        ".": [1,0,0]
    }
    r = np.array(sum([encoding[a] for a in u], []))
    return(r)

def correct_order(u, k=4):
    """Reshapes the one-hot encoding array u to (4, length of string)"""
    return(
        u.reshape((k,int(u.shape[0]/k)), order="f").reshape(u.shape[0])
    )


class CoordPrimaryCapsuleLayer(PrimaryCapsuleLayer):
    """
    A class to define primary capsules with CoordConv
    -------------------------------------------------

    ...

    Attributes
    ----------
    n_capsules : int
        number of capsules
    in_ch : int
        number of input channels
    out_ch : int
        number of output channels
    kernel_size : int or (int, int)
        size of convolutional kernel
    stride : int or (int, int)
        stride of convolution
    use_cuda : bool
        flag specifies whether we should use cuda

    Methods
    -------
    make_conv(self)   
        Build the convolutional layers   
    """
    
    def __init__(
        self, n_capsules=8, in_ch=256, out_ch=32, kernel_size=9, stride=2, use_cuda=True
    ):
        self.use_cuda = use_cuda
        super(CoordPrimaryCapsuleLayer, self).__init__(
            n_capsules, in_ch, out_ch, kernel_size, stride
        )

    def make_conv(self):
        """Builds CoordConv1d kernels according to the specified parameters"""
        return(
            CoordConv1d(
                self.in_ch, self.out_ch, 
                kernel_size=self.kernel_size, 
                stride=self.stride, padding=0, use_cuda=self.use_cuda
            )
        )

    
class OneHotAndCut():
    """
    A class for the preprocessing of gRNAs
    --------------------------------------

    ...

    Attributes
    ----------
    pam : string
        the Protospacer Adjacent Motif sequence
    pam_before : bool
        flag that specifies whether PAM is on 3' or 5' end of the gRNA
    cut_pam : bool
        flag that specifies whether we should use PAM in our final feature vector
    cut_at_start : int
        How much we need to cut out of the flanks 
        at the start of the sequence?
    cut_at_end : int
        How much we need to cut out of the flanks 
        at the end of the sequence?
    fold : bool
        Should we use RNA folding features?
            
            
    Methods
    -------
    __call__(self, x)   
        Run the preprocessing on string x   
    """
    
    def __init__(self, pam, pam_before, cut_pam, cut_at_start=0, cut_at_end=0, fold=False):
        self.PAM = pam
        self.pam_before = pam_before
        self.cut_pam = cut_pam
        self.cut_at_start = cut_at_start
        self.cut_at_end = cut_at_end
        self.fold = fold

    def __call__(self, x):
        """
        Run the preprocessing on string x
    
        Outputs a 1D numpy array with the results of preprocessing
        """
        sequence = x[self.cut_at_start:len(x)-self.cut_at_end]
        if self.cut_pam:
            if self.pam_before:
                sequence = sequence[len(self.PAM)-1:]
            else:
                sequence = sequence[0:-len(self.PAM)+1]
        s_oh = correct_order(onehot(sequence)).reshape(4,len(sequence))
        return(s_oh)


class ToTensor():
    """
    A helper class to make a tensor out ouf preprocessing
    -----------------------------------------------------

    ...

    Attributes
    ----------
    cudap : bool
        controls the usage of cuda            
            
    Methods
    -------
    __call__(self, x)   
        Make a tensor out of x   
    """
    
    def __init__(self, cudap=True):
        self.cuda = torch.cuda.is_available() and cudap

    def __call__(self, x):
        """Make a tensor out of x"""
        r = torch.from_numpy(x).type(torch.FloatTensor)
        if self.cuda:
            r = r.cuda()
        return(r)


class PengDataset(Dataset):
    """
    A torch Dataset for Peng et al. dataset
    ---------------------------------------

    ...

    Attributes
    ----------
    pairs : [string]
        a list of strings of form STRING1,STRING2 with the gRNA-target pairs
    labels : [int]
        a list of 0 or 1 where 0 means no offtarget and 1 means an offtarget
    indices : [int]
        indices to get a subset of the data
    transform : transforms.Compose of a function
        a transformation to apply for each pair
            
    Methods
    -------
    __len__(self)   
        Number of pairs   
    __getitem__(self, ind)   
        Get a pair by its index   
    """
    
    
    def __init__(self, pairs, labels, indices, transform=None):
        self.transform = transform
        self.S = np.array(pairs)[indices]
        self.L = np.array(labels)[indices]

    def __len__(self):
        """Number of pairs"""
        return(len(self.S))

    def sp(self, x, a):
        first = x.split(",")[0] == a.split(",")[0]
        second = x.split(",")[0] not in self.final_neg
        return(first and second)

    def __getitem__(self, ind):
        """
        Get a pair by its index
        
        ind is the index of a pair AFTER indices argument is applied
        
        Outputs a tuple of optionally transformed sequence and target
        """
        sequence = self.S[ind] if not self.transform else self.transform(self.S[ind])
        target = self.L[ind]
        return(sequence, target)


class DeepCRISPRDataset(Dataset):
    """
    A torch Dataset for DeepCRISPR dataset and similar ones
    -------------------------------------------------------

    ...

    Attributes
    ----------
    dataframe : pd.DataFrame
        the dataframe with sequences and labels
    indices : [int]
        indices to get a subset of the data
    n_classes : int
        number of classes to form the labels, -1 specifies regression labels
    transform : transforms.Compose of a function
        a transformation to apply for each pair
    sequence_column : string
        a column name to use as list of sequences
    label_column : string
        a column name to use as list of labels
            
    Methods
    -------
    __len__(self)   
        Number of pairs   
    __getitem__(self, ind)   
        Get a pair by its index   
    label_data(x, u, n)   
        Construct classification labels   
    """
    
    
    def __init__(
        self, dataframe, indices, transform=None, n_classes=-1, 
        sequence_column="sgRNA", label_column="Normalized efficacy"
    ):
        self.transform = transform
        self.n_classes = n_classes
        self.S = dataframe.iloc[indices]
        self.sequence_column = sequence_column
        self.label_column = label_column

    def __len__(self):
        """Number of pairs"""
        return(self.S.shape[0])

    def __getitem__(self, ind):
        """
        Get a pair by its index
        
        ind is the index of a pair AFTER indices argument is applied
        
        Outputs a tuple of optionally transformed sequence and target
        """
        if self.n_classes != -1:
            target = DeepCRISPRDataset.label_data(
                self.S.iloc[ind][self.label_column],
                self.S[self.label_column].values,
                self.n_classes
            )
        else:
            target=self.S.iloc[ind][self.label_column]
        sequence=self.S.iloc[ind][self.sequence_column]
        transformed = self.transform(sequence)
        return(transformed, target)

    @staticmethod
    def label_data(x, u, n):
        """
        Construct classification labels
        
        x is the current value of cleavage efficiency
        u is all values of cleavage efficiency
        n is number of classes
        
        Outputs a class label for x
        """
        p = 100/n
        prc = [(np.percentile(u, p*a), p*a) for a in range(1, n+1)]
        c = n - 1
        for i, a in enumerate(prc):
            if i > 0:
                if x <= prc[i][0] and x > prc[i-1][0]:
                    c = i
                    break
            else:
                if x <= prc[i][0]:
                    c = 0
                    break
        return(c)

    
def plot_LC(tr, te, proportions, description, what="PCC"):
    """
    Plot learning curves
    
    tr is the metrics on training set
    te is the metrics on test set
    proportions is the proportions of the dataset used in the learning curve
    description is the description of the plot
    what specifies what kind of values do we plot
    
    Draws a learning curve
    """
    means_tr = [np.mean(tr[what][a], 0)[0] for a in proportions]
    means_te = [np.mean(te[what][a], 0)[0] for a in proportions]
    mins_tr = [np.min(tr[what][a], 0)[0] for a in proportions]
    mins_te = [np.min(te[what][a], 0)[0] for a in proportions]
    mxs_tr = [np.max(tr[what][a], 0)[0] for a in proportions]
    mxs_te = [np.max(te[what][a], 0)[0] for a in proportions]
    plt.plot(proportions, means_tr, color=(0, 0, 1, 1))
    plt.fill_between(proportions, mins_tr, mxs_tr, color=(0, 0, 1, 0.5))
    plt.plot(proportions, means_te, color=(1, 0, 0, 1))
    plt.fill_between(proportions, mins_te, mxs_te, color=(1, 0, 0, 0.5))
    plt.ylim((0, 1))
    plt.grid(True)
    plt.ylabel(what)
    plt.xlabel("#examples")
    plt.legend(
        handles=[
            Patch(color="blue", label="Train"),
            Patch(color="red", label="Test")
        ], loc="best"
    )
    plt.xticks(*description)


class FastaSet(Dataset):
    """
    A torch Dataset for single record fasta files
    ---------------------------------------------

    ...

    Attributes
    ----------
    fasta : SeqRecord
        contents of fasta file
    pam_regex : string
        regular expression for PAM
    length : int
        length of a gRNA
    transform : transforms.Compose of a function
        a transformation to apply for each pair
    pam_before : bool
        flag that specifies whether PAM is on 3' or 5' end of the gRNA
    use_pam : bool
        flag that specifies whether we should use PAM in our final feature vector
            
        
    Methods
    -------
    get_pam_before(x, s, plen, length)   
        extract gRNA after the PAM   
    get_pam_after(x, s, plen, length)   
        extract gRNA before the PAM   
    getCandidate(start, end)   
        extract gRNA candidate   
    fasta2line(s)   
        strip fasta out of header and join into a single string   
    find_guides(s, pam_regex, length, before=True, strand=1)   
        find available gRNAs   
    __len__(self)   
        Number of gRNAs   
    __getitem__(self, ind)   
        Get a gRNAs by its index   
    """

    def __init__(
        self, fasta, pam_regex, length, pam_before=True,
        transform=None, use_pam=False
    ):
        self.fasta = fasta
        self.T = transform
        self.use_pam = use_pam
        self.length = length
        self.pam_regex = pam_regex
        self.pam_before = pam_before
        line = FastaSet.fasta2line(self.fasta)
        self.guides = FastaSet.find_guides(
            line, self.pam_regex, self.length,
            before=self.pam_before, strand=1
        )
        self.guides.extend(
            FastaSet.find_guides(
                "".join(list(reversed(line))),
                self.pam_regex, self.length,
                before=self.pam_before, strand=-1
            )
        )
        self.guides = list(
            filter(lambda x: len(x[0]) == self.length, self.guides)
        )

    @staticmethod
    def get_pam_before(x, s, plen, length):
        """extract gRNA after the PAM"""
        return(x.extract(s)[0:plen])

    @staticmethod
    def get_pam_after(x, s, plen, length):
        """extract gRNA before the PAM"""
        return(x.extract(s)[length:length+plen])

    @staticmethod
    def getCandidate(start, end):
        """extract gRNA candidate"""
        L = FeatureLocation(start, end)
        f = SeqFeature(L, strand=1, type="sgRNA")
        return(f)

    @staticmethod
    def fasta2line(s):
        """strip fasta out of header and join into a single string"""
        return(
            "".join(
                list(
                    filter(lambda x: not re.match("^>.+$", x), s.split("\n"))
                )
            ).replace(" ", "")
        )

    @staticmethod
    def find_guides(s, pam_regex, length, before=True, strand=1):
        """
        Find available gRNAs
        
        s is the sequence
        pam_regex is the regular expression for PAM
        length is the length of gRNA
        before is true if PAM is on 3' end of gRNA
        strand specifies forward or reverse strands of DNA
        
        Outputs a list of tuples
            (guide, start, end, pam, guide+pam, strand)
        """
        plen = len(pam_regex.replace("[ATGC]", "N"))
        get_pam = FastaSet.get_pam_before if before else FastaSet.get_pam_after
        get_candidates = FastaSet.getCandidate(0, len(s))
        candidates = []
        for a in np.arange(len(s)):
            candidate = (
                int(a), FastaSet.getCandidate(int(a), int(a)+length+plen)
            )
            if re.search(pam_regex, get_pam(candidate[1], s, plen, length)):
                candidates.append(candidate)
        cut_pam = lambda x: x[plen:] if before else x[:-plen]
        guides = []
        for a in candidates:
            guides.append(
                (
                    cut_pam(a[1].extract(s)), a[0], a[0]+length-1,
                    get_pam(a[1], s, plen, length), a[1].extract(s), strand
                )
            )
        return(guides)

    def __len__(self):
        """Number of gRNAs"""
        return(len(self.guides))
    
    def __getitem__(self, ind):
        """
        Get a gRNA by its index
        
        ind is the index of a gRNA
        
        Outputs a tuple
            (guide, start, end, pam, guide+pam, strand, transformed)
        """
        guide, start, end, pam, guidepam, strand = self.guides[ind]
        transformed = self.T(guide) if not self.use_pam else self.T(guidepam)
        return(guide, start, end, pam, guidepam, strand, transformed)


class CasOffFinderSet(Dataset):
    """
    A torch Dataset for CasOffFinder output
    ---------------------------------------

    ...

    Attributes
    ----------
    file : string
        name of the input file
    transform : transforms.Compose of a function
        a transformation to apply for each pair
    only_transformed : bool
        specifies whether we should input all information out of CasOffFinder output or
        only the preprocessed data
        
    Methods
    -------
        
    read_cof_output(file)   
        Parse the output of CasOffFinder   
    __len__(self)   
        Number of gRNAs   
    __getitem__(self, ind)   
        Get a CasOffFinder record by its index   
    """
    
    
    def __init__(self, file, transform=None, only_transformed=False):
        self.D = CasOffFinderSet.read_cof_output(file)
        self.T = transform
        self.only_transformed = only_transformed

    def __getitem__(self, ind):
        """
        Get a CasOffFinder record by its index
        
        ind is the index of the record
        
        if only_transformed is set,
            outputs a tuple of (transformed, index)
        else outputs a tuple
            (guide, title, position, target, strand,
                mismatches, transformed)
        """
        guide = self.D["guide"][ind]
        title = self.D["title"][ind]
        position = self.D["position"][ind]
        target = self.D["target"][ind]
        strand = self.D["strand"][ind]
        mismatches = self.D["mismatches"][ind]
        transformed = self.T(guide+","+target)
        if not self.only_transformed:
            return(
                guide, title, position, target, strand,
                mismatches, transformed
            )
        else:
            return(transformed, np.array([ind]))

    def __len__(self):
        """Number of gRNAs"""
        return(len(self.D["guide"]))

    @staticmethod
    def read_cof_output(file):
        """
        Parse the output of CasOffFinder
        
        file is the filename of the CasOffFinder result
        
        Outputs a dictionary with the parsed information
        """
        u = []
        with open(file, "r") as ih:
            for a in ih:
                L = a.replace("\n", "").split("\t")
                L = list(filter(lambda x: len(x) > 0, L))
                u.append(L)
        u = list(zip(*u))
        d = {
            "guide": u[0], "title": u[1],
            "position": [int(a) for a in u[2]],
            "target": [a.upper() for a in u[3]],
            "strand": [-1 if a == "-" else 1 for a in u[4]],
            "mismatches": [int(a) for a in u[5]]
        }
        return(d)


class GuideHN(nn.Module):
    """
    Guide Hit-or-Miss network with CNN
    ----------------------------------
    
    ...

    Attributes
    ----------
    guide_length : int
        length of gRNA
    capsule_dimension : int
        size of a capsule
    n_routes : int
        number of capsules
    n_classes : int
        number of classes
    n_channels : int
        number of channels for convolutions
    use_cuda : bool
        controls the usage of cuda
        
    Methods
    -------
    forward(self, x)   
        Performs all forward pass computations   
    """
    
    def __init__(
        self, guide_length, capsule_dimension=32,
        n_routes=1280, n_classes=2, n_channels=4,
        use_cuda=False
    ):
        super(GuideHN, self).__init__()
        self.gl = guide_length
        self.use_cuda = use_cuda
        self.n_routes = n_routes
        self.n_classes = n_classes
        self.capsule_dimension = capsule_dimension
        self.conv = nn.Sequential(
            CoordConv1d(
                in_channels=n_channels,
                out_channels=80,
                kernel_size=5,
                stride=1, use_cuda=self.use_cuda
            ),
            nn.LeakyReLU(inplace=True)
        )
        self.hom = HitOrMissLayer(
            in_ch=self.n_routes,
            out_ch=self.capsule_dimension,
            n_classes=self.n_classes
        )
        self.decoder = RegularizingDecoder(
            [
                self.capsule_dimension*self.n_classes,
                512, 1024,
                n_channels*self.gl
            ]
        )

    def forward(self, x):
        """
        Performs all forward pass computations
        
        Outputs internal representations, capsule lengths 
        and reconstructed inputs
        """
        preprocessed = self.conv(x)
        hom = self.hom(
            preprocessed.reshape(
                preprocessed.shape[0], 1, preprocessed.shape[1],
                preprocessed.shape[2]
            )
        )
        internal = 0.5-hom
        lengths = (internal**2).sum(dim=-1)**0.5
        _, max_caps_index = lengths.max(dim=-1)
        masked_internal = mask_hom(hom, max_caps_index.type(torch.FloatTensor))
        reconstructions = self.decoder(
            masked_internal.reshape(masked_internal.shape[0], -1)
        )
        return(internal, lengths, reconstructions)


class GuideHRNN(nn.Module):
    """
    Guide Hit-or-Miss network with RNN
    ----------------------------------

    ...

    Attributes
    ----------
    guide_length : int
        length of gRNA
    capsule_dimension : int
        size of a capsule
    n_routes : int
        number of capsules
    n_classes : int
        number of classes
    n_channels : int
        number of channels for convolutions
    use_cuda : bool
        controls the usage of cuda
        
    Methods
    -------      
    forward(self, x)   
        Performs all forward pass computations   
    """

    def __init__(
        self, guide_length, capsule_dimension=32,
        n_routes=1280, n_classes=2, n_channels=4,
        use_cuda=False
    ):
        super(GuideHRNN, self).__init__()
        self.gl = guide_length
        self.use_cuda = use_cuda
        self.n_routes = n_routes
        self.n_classes = n_classes
        self.capsule_dimension = capsule_dimension
        self.conv = nn.LSTM(
            4, 80, 4, batch_first=True, bidirectional=True
        )
        self.hom = HitOrMissLayer(
            in_ch=self.n_routes,
            out_ch=self.capsule_dimension,
            n_classes=self.n_classes
        )
        self.decoder = RegularizingDecoder(
            [
                self.capsule_dimension*self.n_classes,
                512, 1024,
                n_channels*self.gl
            ]
        )

    def forward(self, x):
        """
        Performs all forward pass computations
        
        Outputs internal representations, capsule lengths 
        and reconstructed inputs
        """
        preprocessed = self.conv(x.permute(0, 2, 1))[0]
        hom = self.hom(
            preprocessed
        )
        internal = 0.5-hom
        lengths = (internal**2).sum(dim=-1)**0.5
        _, max_caps_index = lengths.max(dim=-1)
        masked_internal = mask_hom(
            hom, max_caps_index.type(torch.FloatTensor)
        )
        reconstructions = self.decoder(
            masked_internal.reshape(masked_internal.shape[0], -1)
        )
        return(internal, lengths, reconstructions)

    
class GaussianProcessLayer(DeepGPLayer):
    """
    Gaussian Process Layer
    ----------------------

    ...

    Attributes
    ----------
    input_dims : int
        dimensions of input, the shape of encoder for hidden layer,
        2 (output_dims for hidden layer) for output layer
    output_dims : int
        dimensions of output, 2 for hidden layer, None for output layer
    num_inducing : int
        number of inducing points
    mean_type : string
        mean type, constant mean is for output layer,
        linear mean is for hidden layer
        
    Methods
    -------
    forward(self, x)   
        Performs all forward pass computations   
    """

    def __init__(
        self, input_dims, output_dims,
        num_inducing=128, mean_type="constant"
    ):
        if output_dims is None:
            inducing_points = torch.randn(num_inducing, input_dims)
            batch_shape = torch.Size([])
        else:
            inducing_points = torch.randn(
                output_dims, num_inducing, input_dims
            )
            batch_shape = torch.Size([output_dims])
        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=num_inducing,
            batch_shape=batch_shape
        )
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
        )
        super().__init__(variational_strategy, input_dims, output_dims)
        if mean_type == 'constant':
            self.mean_module = ConstantMean(batch_shape=batch_shape)
        else:
            self.mean_module = LinearMean(input_dims)
        if mean_type == "constant":
            self.covar_module = ScaleKernel(
                RBFKernel(batch_shape=batch_shape, ard_num_dims=input_dims),
                batch_shape=batch_shape, ard_num_dims=None
            )
        else:
            self.covar_module = ScaleKernel(
                RBFKernel(batch_shape=batch_shape, ard_num_dims=input_dims),
                batch_shape=batch_shape, ard_num_dims=None,
                outputscale_constraint=constraints.Interval(0,1)
            )

    def forward(self, x):
        """
        Performs all forward pass computations
        
        Outputs MultivariateNormal with computed mean and covariance
        """
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return(MultivariateNormal(mean, covar))


class DKL(DeepGP):
    """
    Main Deep Kernel Learning module
    --------------------------------

    ...

    Attributes
    ----------
    encoder : model
        A HOM capsule model used as encoder before Gaussian Processes
    train_x_shape : a tuple of ints
        A shape of training set
        
    Methods
    -------
    forward(self, x)   
        Performs all forward pass computations   
    fit(   
        self, train_set_loader, val_set_loaders,   
        epochs, scheduler, optimizer, mll, filename,   
        metric, use_mse=False, use_cuda=True   
    )   
        Train the model   
        
    """

    def __init__(self, encoder, train_x_shape):
        super(DKL, self).__init__()
        hidden_layer = GaussianProcessLayer(
            input_dims=train_x_shape[-1],
            output_dims=2,
            mean_type='linear',
        )
        last_layer = GaussianProcessLayer(
            input_dims=hidden_layer.output_dims,
            output_dims=None,
            mean_type='constant',
        )
        super().__init__()
        self.FE = encoder
        self.hidden_layer = hidden_layer
        self.last_layer = last_layer
        self.likelihood = GaussianLikelihood(
            noise_prior=NormalPrior(0, 0.1),
            noise_constraints=constraints.Positive()
        )
        self.return_encoded = False

    def forward(self, x):
        """
        Performs all forward pass computations
        
        Outputs the prediction and reconstruction if
        return_encoded is not set, otherwise outputs 
        prediction, reconstruction and internal representation
        """
        internal, _, rec = self.FE(x)
        hidden_rep1 = self.hidden_layer(
            internal.reshape(x.shape[0], -1)
        )
        output = self.last_layer(hidden_rep1)
        if self.return_encoded:
            return(output, rec, internal)
        else:
            return(output, rec)

    def fit(
        self, train_set_loader, val_set_loaders,
        epochs, scheduler, optimizer, mll, filename,
        metric, use_mse=False, use_cuda=True
    ):
        """
        Train the model
        
        train_set_loader is torch DataLoader for training set
        val_set_loaders is a list of torch DataLoaders for validation sets
        epochs is the number of epochs to train
        scheduler is the learning rate scheduler
        optimizer is torch.optim optimizer
        mll is MaximumLogLikelihood estimator
        filename is the name of file to put checkpoints in
        metric is the quality measure
        use_mse is whether we should optimize ELBO or ELBO+MSE
        use_cuda specifies the usage of cuda
    
        Outputs training history with train and validation
        """
        training = {"mse": [], "metric": []}
        validation = {
            i: {
                "mse": [], "metric": []
            } for i, a in enumerate(val_set_loaders)
        }
        for a in np.arange(epochs):
            running_targets = []
            running_preds = []
            self.train()
            print("Training, epoch #"+str(a)+", "+str(len(train_set_loader)))
            for i, b in tqdm(enumerate(train_set_loader)):
                sequence, targets = b
                targets = targets.reshape(-1)
                running_targets.append(targets.data.numpy())
                if use_cuda:
                    targets = targets.cuda()
                optimizer.zero_grad()
                output, reconstruction = self.forward(sequence)
                prediction = self.likelihood(output).mean.mean(0)
                loss = -mll(output, targets)
                if use_mse:
                    mse = 0.5*(
                        (self.likelihood(output).mean.mean(0)-targets)**2
                    ).mean()
                    loss = loss + mse
                loss.backward()
                optimizer.step()
                predictions = prediction.cpu().data.numpy()
                running_preds.append(predictions)
            training["mse"].append(
                float(
                    mean_squared_error(
                        np.concatenate(running_preds),
                        np.concatenate(running_targets)
                    )
                )
            )
            training["metric"].append(
                metric(
                    np.concatenate(running_targets),
                    np.concatenate(running_preds).ravel()
                )
            )
            print(
                "Training statistics: "+str(training["mse"][-1]),
                np.unique(np.concatenate(running_preds)).shape,
                str(training["metric"][-1])
            )
            self.eval()
            for j, val_set_loader in enumerate(val_set_loaders):
                print(
                    "Validation, epoch #"+str(
                        a
                    )+", "+str(len(val_set_loader))
                )
                running_targets = []
                running_preds = []
                running_stds = []
                for i, b in tqdm(enumerate(val_set_loader)):
                    sequence, target = b
                    running_targets.append(target.data.numpy())
                    if use_cuda:
                        target = target.cuda()
                    output, _ = self.forward(sequence)
                    loss = -mll(output, target)
                    predictions = self.likelihood(
                        output
                    ).mean.mean(0).cpu().data.numpy()
                    stds = self.likelihood(
                        output
                    ).variance.mean(0).cpu().data.numpy()**0.5
                    running_preds.append(predictions)
                    running_stds.append(stds)
                validation[j]["mse"].append(
                    float(
                        mean_squared_error(
                            np.concatenate(running_preds),
                            np.concatenate(running_targets)
                        )
                    )
                )
                validation[j]["metric"].append(
                    metric(
                        np.concatenate(running_targets),
                        np.concatenate(running_preds).ravel()
                    )
                )
                print(
                    "Validation statistics: "+str(validation[j]["mse"][-1]),
                    np.unique(np.concatenate(running_preds)).shape,
                    str(validation[j]["metric"][-1])
                )
            torch.save(self.state_dict(), filename+str(a)+".ptch")
            scheduler.step()
        return(training, validation)


class JostEtAlDataset(Dataset):
    """
    A torch Dataset for Jost et al. data
    ------------------------------------

    ...

    Attributes
    ----------
    dataframe : pd.DataFrame
        the dataframe with sequences and labels
    indices : [int]
        indices to get a subset of the data
    transform : transforms.Compose of a function
        a transformation to apply for each pair
    sgRNA_column : string
        a column name to use as list of sequences
    label_column : string
        a column name to use as list of labels
    n_bins : int
        a number of bins to weight the samples
    only_target : bool
        specifies whether to return only target or target with bin data
            
    Methods
    -------
    __len__(self)   
        Number of pairs      
    __getitem__(self, ind)   
        Get a pair by its index   
    """

    def __init__(
        self, dataframe, indices, transform=None, genome_column="genome input",
        sgRNA_column="sgRNA input",
        label_column="mean relative gamma", n_bins=5, only_target=True
    ):
        self.transform = transform
        self.S = dataframe.iloc[indices]
        self.genome_column = genome_column
        self.sgRNA_column = sgRNA_column
        self.only_target = only_target
        self.label_column = label_column
        self.n_bins = n_bins
        self.S[self.label_column] = self.S[self.label_column] + np.abs(
            np.min(self.S[self.label_column])
        )
        self.S[self.label_column] = self.S[self.label_column].values/np.max(
            self.S[self.label_column]
        )
        if self.n_bins > 0:
            y_train_clipped = self.S[self.label_column].values.clip(0, 1)
            y_train_binned, histbins = pd.cut(
                y_train_clipped, np.linspace(0, 1, self.n_bins+1),
                labels=range(self.n_bins),
                include_lowest=True, retbins=True
            )
            self.class_weights = {
                k: 1/float(v) for k, v in Counter(y_train_binned).items()
            }
            self.class_weights = {
                k: v/sum(
                    self.class_weights.values()
                ) for k, v in self.class_weights.items()
            }
            self.sample_weights = np.array(
                [self.class_weights[Y] for Y in y_train_binned]
            )

    def __len__(self):
        """Number of pairs"""
        return(self.S.shape[0])

    def __getitem__(self, ind):
        """
        Get a pair by its index
        
        ind is the index of the pair
        
        if only_target is set,
            outputs a tuple of (transformed, target)
        else outputs a tuple
            (transformed, [target, sample_weight])
        """
        guide = self.S.iloc[ind][self.sgRNA_column]
        genome = self.S.iloc[ind][self.genome_column]
        transformed = self.transform(guide+","+genome)
        target = self.S.iloc[ind][self.label_column]
        if self.n_bins > 0 and not self.only_target:
            target = np.array([target, self.sample_weights[ind]])
        return(transformed, target)


class ImperfectMatchTransform():
    """
    Defines the preprocessing for gRNA-target pairs
    -----------------------------------------------

    ...

    Attributes
    ----------
    pam : string
        the Protospacer Adjacent Motif sequence
    pam_before : bool
        flag that specifies whether PAM is on 3' or 5' end of the gRNA
    cut_pam : bool
        flag that specifies whether we should use PAM in our final feature vector
    cut_at_start : int
        How much we need to cut out of the flanks 
        at the start of the sequence?
    cut_at_end : int
        How much we need to cut out of the flanks 
        at the end of the sequence?
    fold : bool
        Should we use RNA folding features?
            
    Methods
    -------
    seq2array(self, x)   
        Convert the sequence to numpy arrays   
    __call__(self, x)      
        Run the preprocessing on string x      
    """

    def __init__(
        self, pam, pam_before, cut_pam, cut_at_start=0,
        cut_at_end=0, fold=False
    ):
        self.PAM = pam
        self.pam_before = pam_before
        self.cut_pam = cut_pam
        self.cut_at_start = cut_at_start
        self.cut_at_end = cut_at_end
        self.fold = fold

    def seq2array(self, x):
        """Convert the sequence to numpy arrays"""
        sequence = x[self.cut_at_start:len(x)-self.cut_at_end]
        if self.cut_pam:
            if self.pam_before:
                sequence = sequence[len(self.PAM)-1:]
            else:
                sequence = sequence[0:-len(self.PAM)+1]
        s_oh = correct_order(onehot(sequence)).reshape(4, len(sequence))
        if self.fold:
            fold, energy = RNA.fold(sequence)
            f_oh = correct_order(
                onehot(fold), k=3
            ).reshape(3, len(sequence))
            s_oh = np.concatenate([s_oh, f_oh])
        return(s_oh)

    def __call__(self, x):
        """
        Run the preprocessing on pair x
    
        Outputs a (2,4,N) numpy array with the results of preprocessing
        """
        guide, genome = x.split(",")
        guide = self.seq2array(guide)
        genome = self.seq2array(genome)
        return(np.stack([guide, genome]))


class GuideHN2d(nn.Module):
    """
    Guide Hit-or-Miss network with 2D-CNN
    -------------------------------------

    ...

    Attributes
    ----------
    guide_length : int   
        length of gRNA   
    capsule_dimension : int   
        size of a capsule   
    n_routes : int   
        number of capsules   
    n_classes : int   
        number of classes   
    n_channels : int   
        number of channels for convolutions   
    use_cuda : bool   
        controls the usage of cuda   
      
    Methods
    -------
    forward(self, x)   
        Performs all forward pass computations   
    """

    def __init__(
        self, guide_length, capsule_dimension=32,
        n_routes=1280, n_classes=2, n_channels=4,
        use_cuda=False
    ):
        super(GuideHN2d, self).__init__()
        self.gl = guide_length
        self.use_cuda = use_cuda
        self.n_routes = n_routes
        self.n_classes = n_classes
        self.capsule_dimension = capsule_dimension
        self.conv = nn.Sequential(
            CoordConv2d(
                in_channels=n_channels,
                out_channels=80,
                kernel_size=(4, 4),
                stride=1, use_cuda=self.use_cuda
            ),
            nn.LeakyReLU(inplace=True)
        )
        self.hom = HitOrMissLayer(
            in_ch=self.n_routes,
            out_ch=self.capsule_dimension,
            n_classes=self.n_classes
        )
        self.decoder = RegularizingDecoder(
            [
                self.capsule_dimension*self.n_classes,
                512, 1024,
                n_channels*self.gl*4
            ]
        )

    def forward(self, x):
        """
        Performs all forward pass computations
        
        Outputs internal representations, capsule lengths 
        and reconstructed inputs
        """
        preprocessed = self.conv(x)
        hom = self.hom(
            preprocessed.permute([0, 2, 1, 3])
        )
        internal = 0.5-hom
        lengths = (internal**2).sum(dim=-1)**0.5
        _, max_caps_index = lengths.max(dim=-1)
        masked_internal = mask_hom(hom, max_caps_index.type(torch.FloatTensor))
        reconstructions = self.decoder(
            masked_internal.reshape(masked_internal.shape[0], -1)
        )
        return(internal, lengths, reconstructions)


class DeepHFDataset(Dataset):
    """
    A torch Dataset for DeepHF data
    -------------------------------

    ...

    Attributes
    ----------
    dataframe : pd.DataFrame
        the dataframe with sequences and labels
    indices : [int]
        indices to get a subset of the data
    transform : transforms.Compose of a function
        a transformation to apply for each pair
    sequence_column : string
        a column name to use as list of sequences
    label_column : string
        a column name to use as list of labels
            
    Methods
    -------
    __len__(self)   
        Number of gRNAs   
    __getitem__(self, ind)   
        Get a gRNA by its index   
    """

    def __init__(
        self, dataframe, indices, transform=None,
        sequence_column="sgRNA", label_column="Normalized efficacy",
        labelling="Smart"
    ):
        self.transform = transform
        self.S = dataframe.iloc[indices]
        self.sequence_column = sequence_column
        self.label_column = label_column
        self.labelling = labelling

    def __len__(self):
        """Number of gRNAs"""
        return(self.S.shape[0])

    def __getitem__(self, ind):
        """
        Get a gRNA by its index
        
        ind is the index of the gRNA
        
        Outputs a tuple of (transformed, target)
        """
        target = self.S.iloc[ind][self.label_column]
        sequence = self.S.iloc[ind][self.sequence_column]
        transformed = self.transform(sequence)
        return(transformed, target)


class GeCRISPRDataset(Dataset):
    """
    A torch Dataset for geCRISPR data
    ---------------------------------

    ...

    Attributes
    ----------
    file : string
        the file name for geCRISPR data
    indices : [int]
        indices to get a subset of the data
    transform : transforms.Compose of a function
        a transformation to apply for each pair
    classification : bool
        output classes or real values
            
    Methods
    -------
    __len__(self)   
        Number of gRNAs   
    __getitem__(self, ind)   
        Get a gRNA by its index    
    """

    def __init__(self, file, indices, transform=None, classification=True):
        self.transform = transform
        self.classification = classification
        self.S = pd.read_csv(file, sep=" ", header=None).iloc[indices]

    def __len__(self):
        """Number of gRNAs"""
        return(self.S.shape[0])

    def __getitem__(self, ind):
        """
        Get a gRNA by its index
        
        ind is the index of the gRNA
        
        Outputs a tuple of (transformed, target)
        target is one of -1 or 1 if classification flag is set,
        a value between 0 and 1 otherwise
        """
        if self.classification:
            target = [-1, 1].index(self.S.iloc[ind][0])
        else:
            target = self.S.iloc[ind][0]/100
        sequence = self.S.iloc[ind][1]
        transformed = self.transform(sequence)
        return(transformed, target)
