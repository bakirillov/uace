#!/usr/bin/env python
# coding: utf-8


# ### Necessary imports

# In[1]:


import re
import os
import math
import torch
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
from IPython.display import Image
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from torchvision import models, transforms
from scipy.stats import pearsonr, spearmanr
from gpytorch.priors import SmoothedBoxPrior
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
from gpytorch.kernels import ScaleKernel, RBFKernel, GridInterpolationKernel
from gpytorch.mlls import VariationalELBO, VariationalELBOEmpirical, DeepApproximateMLL
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error


def in_CI(ys, y_hats, stds):
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
    return(float(pearsonr(a, b)[0]**2))


def iterate_minibatches(X, y, batchsize, permute=False):
    indices = np.random.permutation(np.arange(len(X))) if permute else np.arange(len(X))
    for start in range(0, len(indices), batchsize):
        ix = indices[start: start + batchsize]
        yield X[ix], y[ix]


def moving_average(net1, net2, alpha=1):
    """Moving average over weights as described in the SWA paper"""
    for param1, param2 in zip(net1.parameters(), net2.parameters()):
        param1.data *= (1.0 - alpha)
        param1.data += param2.data * alpha



def onehot(u):
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
    return(
        u.reshape((k,int(u.shape[0]/k)), order="f").reshape(u.shape[0])
    )



class CoordPrimaryCapsuleLayer(PrimaryCapsuleLayer):
    
    def __init__(
        self, n_capsules=8, in_ch=256, out_ch=32, kernel_size=9, stride=2, use_cuda=True
    ):
        self.use_cuda = use_cuda
        super(CoordPrimaryCapsuleLayer, self).__init__(
            n_capsules, in_ch, out_ch, kernel_size, stride
        )
        
    def make_conv(self):
        return(
            CoordConv1d(
                self.in_ch, self.out_ch, 
                kernel_size=self.kernel_size, 
                stride=self.stride, padding=0, use_cuda=self.use_cuda
            )
        )
    
    
class Coord2dPrimaryCapsuleLayer(PrimaryCapsuleLayer):
    
    def __init__(
        self, n_capsules=8, in_ch=256, out_ch=32, kernel_size=9, stride=2, use_cuda=True
    ):
        self.use_cuda = use_cuda
        super(Coord2dPrimaryCapsuleLayer, self).__init__(
            n_capsules, in_ch, out_ch, kernel_size, stride
        )
        
    def make_conv(self):
        return(
            CoordConv2d(
                self.in_ch, self.out_ch, 
                kernel_size=self.kernel_size, 
                stride=self.stride, padding=0,
                use_cuda=self.use_cuda
            )
        )
    
    
class LinearPrimaryCapsuleLayer(PrimaryCapsuleLayer):
    
    def __init__(
        self, n_capsules=8, in_ch=256, out_ch=32, kernel_size=9, stride=2, dropout=0.5
    ):
        self.dropout = dropout
        super(LinearPrimaryCapsuleLayer, self).__init__(
            n_capsules, in_ch, out_ch, kernel_size, stride
        )
        
    def make_conv(self):
        return(
            nn.Sequential(
                nn.Linear(self.in_ch, self.out_ch),
                nn.Dropout(self.dropout)
            )
        )
    

class GRUPrimaryCapsuleLayer(PrimaryCapsuleLayer):
    
    def make_conv(self):
        """Build a primary capsule which is just single layer LSTM"""
        return(
            nn.GRU(
                self.in_ch, self.out_ch, num_layers=self.n_layers, 
                dropout=self.dropout, batch_first=True
            )
        )
    
    def __init__(
        self, n_capsules=8, in_ch=256, out_ch=32, dropout=0, n_layers=4
    ):
        self.dropout = dropout
        self.n_layers = n_layers
        super(GRUPrimaryCapsuleLayer, self).__init__(
            n_capsules, in_ch, out_ch, 9, 2
        )
            
    def forward(self, x):
        """Compute outputs of capsules, reshape and squash"""
        out = torch.cat(
            [
                a(x.reshape(x.shape[0],x.shape[1],1))[0].contiguous().view(x.size(0), -1, 1) 
                    for a in self.capsules
            ], 
            dim=-1
        )
        return(squash(out))
    
    
class DualLine(nn.Module):
    
    def __init__(self):
        self.alpha = nn.Parameter(
            torch.randn()
        )
        self.beta = nn.Parameter(
            torch.randn()
        )
        self.mu = nn.Parameter(-0.22)
        
    def once(self, x):
        return(
            torch.cat([self.alpha*a+self.mu if a < 0 else self.beta*a + self.mu for a in x])
        )
        
    def forward(self, x):
        y = torch.stack([self.once(a) for a in x])
        return(y)

    
class GuideCaps(nn.Module):
    
    def __init__(
        self, guide_length, n_routes=120, n_classes=2, n_prim_capsules=128, n_iter=3,
        use_deconv=False, rec_ks=5, rec_pad=8, use_cuda=True
    ):
        super(GuideCaps, self).__init__()
        self.gl = guide_length
        self.n_routes = n_routes
        self.n_classes = n_classes
        self.n_iter = n_iter
        self.use_cuda = use_cuda
        self.n_prim_capsules = n_prim_capsules
        self.conv = nn.Sequential(
            CoordConv1d(
                in_channels=4, 
                out_channels=80,
                kernel_size=5,
                stride=1, use_cuda=self.use_cuda
            ),
            nn.LeakyReLU(inplace=True)
        )
        self.primcaps = CoordPrimaryCapsuleLayer(
            n_capsules=self.n_prim_capsules, in_ch=80, out_ch=16, kernel_size=2,
            stride=1, use_cuda=self.use_cuda
        )
        self.classcaps = SecondaryCapsuleLayer(
            n_capsules=self.n_classes, n_iter=self.n_iter, 
            in_ch=self.n_prim_capsules, out_ch=32, n_routes=self.n_routes,
            cuda=self.use_cuda
        )
        self.deconv = use_deconv
        if not self.deconv:
            self.decoder = RegularizingDecoder(
                dims=[32*self.n_classes, 128, 256, 4*self.gl]
            )
        else:
            self.decoder = nn.ConvTranspose1d(
                in_channels=2, out_channels=4, kernel_size=rec_ks, padding=rec_pad
            )
        
    def forward(self, x):
        co = self.conv(x)
        pc = self.primcaps(co)
        internal = self.classcaps(pc)
        lengths = (internal**2).sum(dim=-1)**0.5
        _, max_caps_index = lengths.max(dim=-1)
        masked = torch.eye(self.n_classes)
        masked = masked.cuda() if torch.cuda.is_available() and self.use_cuda else masked
        masked = masked.index_select(dim=0, index=max_caps_index.data)
        masked_internal = (internal*masked[:,:,None])
        if not self.deconv:
            masked_internal = masked_internal.view(x.size(0), -1)
        reconstruction = self.decoder(masked_internal)
        return(internal, reconstruction, lengths, max_caps_index, masked_internal)
    
    
class OneHotAndCut():
    
    def __init__(self, pam, pam_before, cut_pam, cut_at_start=0, cut_at_end=0, fold=False):
        self.PAM = pam
        self.pam_before = pam_before
        self.cut_pam = cut_pam
        self.cut_at_start = cut_at_start
        self.cut_at_end = cut_at_end
        self.fold = fold
        
    def __call__(self, x):
        sequence = x[self.cut_at_start:len(x)-self.cut_at_end]
        if self.cut_pam:
            if self.pam_before:
                sequence = sequence[len(self.PAM)-1:]
            else:
                sequence = sequence[0:-len(self.PAM)+1]
        s_oh = correct_order(onehot(sequence)).reshape(4,len(sequence))
        if self.fold:
            fold, energy = RNA.fold(sequence)
            f_oh = correct_order(onehot(fold), k=3).reshape(3, len(sequence))
           # energy = np.repeat(energy, 4+3).reshape(7,1)
            s_oh = np.concatenate([s_oh, f_oh])#, energy], 1)
        return(s_oh)          
                            
                            
class BinaryMismatches():
    
    def __init__(
        self, pam, pam_before, cut_pam, cut_at_start=0, cut_at_end=0, recurrent=False,
        use_peng_additions=False
    ):
        self.PAM = pam
        self.pam_before = pam_before
        self.cut_pam = cut_pam
        self.cut_at_start = cut_at_start
        self.cut_at_end = cut_at_end
        self.recurrent = recurrent
        self.use_peng_additions = use_peng_additions
        self.scheme = {
            "AT": 0, "AC": 1, "AG": 2,
            "TA": 3, "TC": 4, "TG": 5,
            "CA": 6, "CT": 7, "CG": 8,
            "GA": 9, "GT": 10, "GC": 11,
        }
        
    def process(self, x):
        sequence = x[self.cut_at_start:len(x)-self.cut_at_end]
        if self.cut_pam:
            if self.pam_before:
                sequence = sequence[len(self.PAM)-1:]
            else:
                sequence = sequence[0:-len(self.PAM)+1]
        return(sequence)
        
    @staticmethod
    def count_gc(s):
        s = s.upper()
        gc = s.count("G")+s.count("C")
        at = s.count("A")+s.count("T")
        gc_c = gc/len(s)
        gc_skew = 0 if gc == 0 else (s.count("G")-s.count("C"))/gc
        at_skew = 0 if at == 0 else (s.count("A")-s.count("T"))/at
        rat = 0 if at_skew == 0 else gc_skew/at_skew
        return(np.array([gc, gc_c, gc_skew, at_skew, rat]))
        
    @staticmethod
    def compute_peng_additions(s1, s2):
        ad1 = BinaryMismatches.count_gc(s1)
        ad2 = BinaryMismatches.count_gc(s2)
        return(ad2-ad1)
        
    def __call__(self, x):
        sequence1, sequence2 = list(map(lambda u: self.process(u), x.split(",")))
        assert len(sequence1)==len(sequence2)
        out = np.zeros(shape=(12,len(sequence1)))
        for i,(a,b) in enumerate(zip(sequence1, sequence2)):
            if a != b:
                out[self.scheme[a+b], i] += 1
        if self.recurrent:
            out = out.reshape(-1)
            if self.use_peng_additions:
                adds = BinaryMismatches.compute_peng_additions(sequence1, sequence2)
                out = np.concatenate([out, adds])
        return(out)
    
    
class ToTensor():
    
    def __init__(self, cudap=True):
        self.cuda = torch.cuda.is_available() and cudap
    
    def __call__(self, x):
        r = torch.from_numpy(x).type(torch.FloatTensor)
        if self.cuda:
            r = r.cuda()
        return(r)
    
    
class PengDataset(Dataset):
    
    def __init__(self, pairs, labels, indices, transform=None):
        self.transform = transform
        self.S = np.array(pairs)[indices]
        self.L = np.array(labels)[indices]
    
    def __len__(self):
        return(len(self.S))
    
    def sp(self, x, a):
        first = x.split(",")[0] == a.split(",")[0]
        second = x.split(",")[0] not in self.final_neg
        return(first and second)
    
    def __getitem__(self, ind):
        sequence = self.S[ind] if not self.transform else self.transform(self.S[ind])
        target = self.L[ind]
        return(sequence, target)
    
    
class DeepCRISPRDataset(Dataset):
    
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
        return(self.S.shape[0])
    
    def __getitem__(self, ind):
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
        p = 100/n
        prc = [(np.percentile(u, p*a), p*a) for a in range(1,n+1)]
        c = n-1
        for i,a in enumerate(prc):
            if i > 0:
                if x <= prc[i][0] and x > prc[i-1][0]:
                    c = i
                    break
            else:
                if x <= prc[i][0]:
                    c = 0
                    break
        return(c)
    
    
class OneHotCutDiff():
    
    def __init__(self, pam, pam_before, cut_pam):
        self.PAM = pam
        self.pam_before = pam_before
        self.cut_pam = cut_pam
        self.ohc = OneHotAndCut(self.PAM, self.pam_before, self.cut_pam)
        
    def __call__(self, x):
        sequence = x
        guide, target = x.split(",")
        guide = self.ohc(guide)
        target = self.ohc(target)
        difference = guide - target
        return(difference)
    
            
def plot_LC(tr, te, proportions, description, what="PCC"):
    means_tr = [np.mean(tr[what][a], 0)[0] for a in proportions]
    means_te = [np.mean(te[what][a], 0)[0] for a in proportions]
    mins_tr = [np.min(tr[what][a], 0)[0] for a in proportions]
    mins_te = [np.min(te[what][a], 0)[0] for a in proportions]
    mxs_tr = [np.max(tr[what][a], 0)[0] for a in proportions]
    mxs_te = [np.max(te[what][a], 0)[0] for a in proportions]
    plt.plot(proportions, means_tr, color=(0,0,1,1))
    plt.fill_between(proportions, mins_tr, mxs_tr, color=(0,0,1,0.5))
    plt.plot(proportions, means_te, color=(1,0,0,1))
    plt.fill_between(proportions, mins_te, mxs_te, color=(1,0,0,0.5))
    plt.ylim((0,1))
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
            line, self.pam_regex, self.length, before=self.pam_before, strand=1
        )
        self.guides.extend(
            FastaSet.find_guides(
                "".join(list(reversed(line))), self.pam_regex, self.length,
                before=self.pam_before, strand=-1
            )
        )
        self.guides = list(filter(lambda x: len(x[0]) == self.length, self.guides))
        
    @staticmethod
    def get_pam_before(x, s, plen, length):
        return(x.extract(s)[0:plen])
    
    @staticmethod
    def get_pam_after(x, s, plen, length):
        return(x.extract(s)[length:length+plen])
    
    @staticmethod
    def getCandidate(start, end):
        l = FeatureLocation(start, end)
        f = SeqFeature(l, strand=1, type="sgRNA")
        return(f)
        
    @staticmethod
    def fasta2line(s):
        return(
            "".join(list(filter(lambda x: not re.match("^>.+$", x), s.split("\n")))).replace(" ", "")
        )
        
    @staticmethod
    def find_guides(s, pam_regex, length, before=True, strand=1):
        plen = len(pam_regex.replace("[ATGC]", "N"))
        get_pam = FastaSet.get_pam_before if before else FastaSet.get_pam_after
        get_candidates = FastaSet.getCandidate(0, len(s))
        candidates = []
        for a in np.arange(len(s)):
            candidate = (int(a), FastaSet.getCandidate(int(a), int(a)+length+plen))
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
        return(len(self.guides))
    
    def __getitem__(self, ind):
        guide, start, end, pam, guidepam, strand = self.guides[ind]
        transformed = self.T(guide) if not self.use_pam else self.T(guidepam)
        return(guide, start, end, pam, guidepam, strand, transformed)
    
    
class CasOffFinderSet(Dataset):
    
    def __init__(self, file, transform=None, only_transformed=False):
        self.D = CasOffFinderSet.read_cof_output(file)
        self.T = transform
        self.only_transformed = only_transformed
        
    def __getitem__(self, ind):
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
        return(len(self.D["guide"]))
    
    @staticmethod
    def read_cof_output(file):
        u = []
        with open(file, "r") as ih:
            for a in ih:
                l = a.replace("\n", "").split("\t")
                l = list(filter(lambda x: len(x) > 0, l))
                u.append(l)
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
    
    def __init__(
        self, guide_length, capsule_dimension=32, n_routes=1280, n_classes=2, n_channels=4,
        use_cuda=True
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
        self.hom = HitOrMissLayer(#AttentionHOMLayer(
            in_ch=self.n_routes, 
            out_ch=self.capsule_dimension, 
            n_classes=self.n_classes
        )
        self.decoder = RegularizingDecoder(
            [
                self.capsule_dimension*self.n_classes,
                512,1024,
                n_channels*self.gl
            ]
        )
    
    def forward(self, x):
        preprocessed = self.conv(x)
        hom = self.hom(
            preprocessed.reshape(
                preprocessed.shape[0], 1, preprocessed.shape[1], preprocessed.shape[2]
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
    
    def __init__(
        self, guide_length, capsule_dimension=32, n_routes=1280, n_classes=2, n_channels=4,
        use_cuda=True
    ):
        super(GuideHRNN, self).__init__()
        self.gl = guide_length
        self.use_cuda = use_cuda
        self.n_routes = n_routes
        self.n_classes = n_classes
        self.capsule_dimension = capsule_dimension
        self.conv = nn.LSTM(4, 80, 4, batch_first=True, bidirectional=True)
        self.hom = HitOrMissLayer(
            in_ch=self.n_routes, 
            out_ch=self.capsule_dimension, 
            n_classes=self.n_classes
        )
        self.decoder = RegularizingDecoder(
            [
                self.capsule_dimension*self.n_classes,
                512,1024,
                n_channels*self.gl
            ]
        )
    
    def forward(self, x):
        preprocessed = self.conv(x.permute(0,2,1))[0]
        hom = self.hom(
            preprocessed
        )
        internal = 0.5-hom
        lengths = (internal**2).sum(dim=-1)**0.5
        _, max_caps_index = lengths.max(dim=-1)
        masked_internal = mask_hom(hom, max_caps_index.type(torch.FloatTensor))
        reconstructions = self.decoder(
            masked_internal.reshape(masked_internal.shape[0], -1)
        )
        return(internal, lengths, reconstructions)
    
    
class OfftargetHOM(nn.Module):
    
    def __init__(
        self, guide_length=23, addition_length=5, 
        capsule_dimension=32, n_classes=2
    ):
        super(OfftargetHOM, self).__init__()
        self.guide_length = guide_length
        self.addition_length = addition_length
        self.n_classes = n_classes
        self.capsule_dimension = capsule_dimension
        hom_input_length = self.guide_length*12#+self.addition_length
        self.preprocess = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=64, kernel_size=(3,3), stride=1
            ),
            nn.ReLU(),
            models.resnet.BasicBlock(64, 64),
            models.resnet.BasicBlock(64, 64),
            models.resnet.BasicBlock(64, 64),
            models.resnet.BasicBlock(64, 64)
        )
        self.hom = HitOrMissLayer(
            in_ch=13440, 
            out_ch=self.capsule_dimension, 
            n_classes=self.n_classes
        )
        self.decoder = RegularizingDecoder(
            [
                self.capsule_dimension*self.n_classes,
                512,1024,
                hom_input_length
            ]
        )

    def forward(self, x):
        preprocessed = self.preprocess(x[:,:-5].reshape(x.shape[0], 1, 12, 23))
        hom = self.hom(preprocessed)
        internal = 0.5-hom
        lengths = (internal**2).sum(dim=-1)**0.5
        _, max_caps_index = lengths.max(dim=-1)
        masked_internal = mask_hom(hom, max_caps_index.type(torch.FloatTensor))
        reconstructions = self.decoder(
            masked_internal.reshape(masked_internal.shape[0], -1)
        )
        return(internal, lengths, reconstructions)
    
    
class GaussianProcessLayer(DeepGPLayer):
    
    def __init__(self, input_dims, output_dims, num_inducing=128, mean_type="constant"):
        if output_dims is None:
            inducing_points = torch.randn(num_inducing, input_dims)
            batch_shape = torch.Size([])
        else:
            inducing_points = torch.randn(output_dims, num_inducing, input_dims)
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
        self.covar_module = ScaleKernel(
            RBFKernel(batch_shape=batch_shape, ard_num_dims=input_dims),
            batch_shape=batch_shape, ard_num_dims=None
        )
        
    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return(MultivariateNormal(mean, covar))
    
    
class DKL(DeepGP):
    
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
        self.likelihood = GaussianLikelihood()
        self.return_encoded = False
    
    def forward(self, x):
        internal, _, rec = self.FE(x)
        hidden_rep1 = self.hidden_layer(internal.reshape(x.shape[0], -1))#x.reshape(x.shape[0], -1))
        output = self.last_layer(hidden_rep1)
        if self.return_encoded:
            return(output, rec, internal)
        else:
            return(output, rec)
        
    def fit(self, train_set_loader, val_set_loader, epochs, scheduler, optimizer, mll, filename, metric):
        training = {"mse": [], "metric": []}
        validation = {"mse": [], "metric": []}
        for a in np.arange(epochs):
            running_targets = []
            running_preds = []
            self.train()
            print("Training, epoch #"+str(a)+", "+str(len(train_set_loader)))
            for i,b in tqdm(enumerate(train_set_loader)):
                sequence, targets = b
                running_targets.append(targets[:,0].data.numpy())
                targets = targets.cuda()
                optimizer.zero_grad()
                output, reconstruction = self.forward(sequence)
                prediction = self.likelihood(output).mean.mean(0)
                loss = -mll(output, targets[:,0])
                loss.backward()
                optimizer.step()
                predictions = prediction.cpu().data.numpy()
                running_preds.append(predictions)
            training["mse"].append(
                float(mean_squared_error(np.concatenate(running_preds), np.concatenate(running_targets)))
            )
            training["metric"].append(
                metric(np.concatenate(running_targets), np.concatenate(running_preds).ravel())
            )
            print(
                "Training statistics: "+str(training["mse"][-1]),
                np.unique(np.concatenate(running_preds)).shape, str(training["metric"][-1])
            )
            self.eval()
            running_targets = []
            running_preds = []
            print("Validation, epoch #"+str(a)+", "+str(len(val_set_loader)))
            for i,b in tqdm(enumerate(val_set_loader)):
                sequence, target = b
                running_targets.append(target.data.numpy())
                target = target.cuda()
                output, _ = self.forward(sequence)
                loss = -mll(output, target)
                predictions = self.likelihood(output).mean.mean(0).cpu().data.numpy()
                running_preds.append(predictions)
            validation["mse"].append(
                float(mean_squared_error(np.concatenate(running_preds), np.concatenate(running_targets)))
            )
            validation["metric"].append(
                metric(np.concatenate(running_targets), np.concatenate(running_preds).ravel())
            )
            torch.save(self.state_dict(), filename+str(a)+".ptch")
            print(
                "Validation statistics: "+str(validation["mse"][-1]),
                np.unique(np.concatenate(running_preds)).shape, str(validation["metric"][-1])
            )
            scheduler.step()
        return(training, validation)
        
        
class WeissmanDataset(Dataset):
    
    def __init__(
        self, dataframe, indices, transform=None, genome_column="genome input", sgRNA_column="sgRNA input",
        label_column="mean relative gamma", n_bins=5
    ):
        self.transform = transform
        self.S = dataframe.iloc[indices]
        self.genome_column = genome_column
        self.sgRNA_column = sgRNA_column
        self.label_column = label_column
        self.n_bins = n_bins
        self.S[self.label_column] = self.S[self.label_column] + np.abs(np.min(self.S[self.label_column]))
        self.S[self.label_column] = self.S[self.label_column].values/np.max(self.S[self.label_column])
        if self.n_bins > 0:
            y_train_clipped = self.S[self.label_column].values.clip(0,1)
            y_train_binned, histbins = pd.cut(
                y_train_clipped, np.linspace(0,1,self.n_bins+1), labels=range(self.n_bins), 
                include_lowest=True, retbins=True
            )
            self.class_weights = {k:1/float(v) for k,v in Counter(y_train_binned).items()}
            self.class_weights = {k:v/sum(self.class_weights.values()) for k,v in self.class_weights.items()}
            self.sample_weights = np.array([self.class_weights[Y] for Y in y_train_binned])
        
        
    def __len__(self):
        return(self.S.shape[0])
    
    def __getitem__(self, ind):
        guide = self.S.iloc[ind][self.sgRNA_column]
        genome = self.S.iloc[ind][self.genome_column]
        transformed = self.transform(guide+","+genome)
        target = self.S.iloc[ind][self.label_column]
        if self.n_bins > 0:
            target = np.array([target, self.sample_weights[ind]])
        return(transformed, target)
    
    
class ImperfectMatchTransform():
    
    def __init__(self, pam, pam_before, cut_pam, cut_at_start=0, cut_at_end=0, fold=False):
        self.PAM = pam
        self.pam_before = pam_before
        self.cut_pam = cut_pam
        self.cut_at_start = cut_at_start
        self.cut_at_end = cut_at_end
        self.fold = fold
        
    def seq2array(self, x):
        sequence = x[self.cut_at_start:len(x)-self.cut_at_end]
        if self.cut_pam:
            if self.pam_before:
                sequence = sequence[len(self.PAM)-1:]
            else:
                sequence = sequence[0:-len(self.PAM)+1]
        s_oh = correct_order(onehot(sequence)).reshape(4,len(sequence))
        if self.fold:
            fold, energy = RNA.fold(sequence)
            f_oh = correct_order(onehot(fold), k=3).reshape(3, len(sequence))
           # energy = np.repeat(energy, 4+3).reshape(7,1)
            s_oh = np.concatenate([s_oh, f_oh])#, energy], 1)
        return(s_oh)
        
    def __call__(self, x):
        guide, genome = x.split(",")
        guide = self.seq2array(guide)
        genome = self.seq2array(genome)
        return(np.stack([guide, genome]))
    
    
class GuideHN2d(nn.Module):
    
    def __init__(
        self, guide_length, capsule_dimension=32, n_routes=1280, n_classes=2, n_channels=4,
        use_cuda=True
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
                kernel_size=(4,4),
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
                512,1024,
                n_channels*self.gl*4
            ]
        )
    
    def forward(self, x):
        preprocessed = self.conv(x)
        hom = self.hom(
            preprocessed.permute([0,2,1,3])
        )
        internal = 0.5-hom
        lengths = (internal**2).sum(dim=-1)**0.5
        _, max_caps_index = lengths.max(dim=-1)
        masked_internal = mask_hom(hom, max_caps_index.type(torch.FloatTensor))
        reconstructions = self.decoder(
            masked_internal.reshape(masked_internal.shape[0], -1)
        )
        return(internal, lengths, reconstructions)