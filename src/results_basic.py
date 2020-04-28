#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os
import json
import argparse
import numpy as np
import pandas as pd
import os.path as op
from core import in_CI, rsquared
from scipy.stats import spearmanr, pearsonr


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input",
        dest="input",
        action="store", 
        help="set the mask of input"
    )
    parser.add_argument(
        "-o", "--output",
        dest="output",
        action="store", 
        help="set the path of output directory"
    )
    args = parser.parse_args()
    with open(args.input+".json", "r") as ih:
        data = json.load(ih)
    CI = in_CI(data["y"], data["y_hat"], data["y_hat_std"]).mean()
    pearson = pearsonr(data["y"], data["y_hat"])
    spearman = spearmanr(data["y"], data["y_hat"])
    rsq = rsquared(data["y"], data["y_hat"])
    results = pd.DataFrame(CI).T
    results["PCC"] = [pearson[0]]
    results["PCC-pval"] = [pearson[1]]
    results["SCC"] = [spearman[0]]
    results["SCC-pval"] = [spearman[1]]
    results["rsquared"] = [rsq]
    outfn = op.join(args.output, args.input.replace("/", "_").replace(".", ""))
    print(outfn)
    results.to_csv(outfn+".csv")