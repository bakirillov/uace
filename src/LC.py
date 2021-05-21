import os
import argparse
import os.path as op


templates = {
    "DeepCpf1Kim": "python reproduce_DeepCpf1_Kim.py -m CNN -s SEED -p PROP -o OUTPUT_PROP_",
    "DeepHFWt": "python reproduce_DeepHF.py -d WT -m CNN -s SEED -p PROP -o OUTPUT_PROP_",
    "DeepHFeSpCas9": "python reproduce_DeepHF.py -d eSpCas9 -m CNN -s SEED -p PROP -o OUTPUT_PROP_",
    "DeepHFSpCas9HF1": "python reproduce_DeepHF.py -d SpCas9HF1 -m CNN -s SEED -p PROP -o OUTPUT_PROP_",
    "Cas9_Offtarget": "python reproduce_2d_models.py -s SEED -p PROP -o OUTPUT_cnn_elbo_PROP_",
    "Cpf1_Offtarget": "python reproduce_2d_models.py -d Cpf1 -s SEED -p PROP -o OUTPUT_cnn_elbo_PROP_",
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--dataset",
        dest="dataset",
        action="store",
        help="the dataset"
    )
    parser.add_argument(
        "-p", "--proportions",
        dest="proportions",
        action="store",
        default="0.05,0.1,0.2,0.3,0.4,0.5,0.75,0.9,0.95,0.99",
        help="set the proportions"
    )
    parser.add_argument(
        "-o", "--output",
        dest="output",
        action="store",
        help="set the path of output directory"
    )
    parser.add_argument(
        "-f", "--file",
        dest="file",
        action="store",
        help="set the path of output file"
    )
    parser.add_argument(
        "-s", "--seed",
        dest="seed",
        action="store",
        help="set the seed for prng",
        default=192
    )
    args = parser.parse_args()
    if not op.exists(args.output):
        os.makedirs(args.output)
    T = templates[args.dataset]
    T = T.replace("OUTPUT", args.output)
    proportions = args.proportions.split(",")
    script = open(args.file, "w")
    script.write("#!/bin/sh\n\n\n")
    for b in proportions:
        current = T.replace("PROP", str(b)).replace("SEED", str(args.seed))
        script.write(current+"\n")
    script.write("\n")
    for b in proportions:
        current = T.replace(
            "PROP", str(b)
        ).replace(
            "elbo", "mse"
        ).replace("SEED", str(args.seed))+" -u"
        script.write(current+"\n")
    script.write("\n")
    if "Offtarget" not in T:
        for b in proportions:
            current = T.replace(
                "PROP", str(b)
            ).replace(
                "CNN", "RNN"
            ).replace(
                "cnn", "rnn"
            ).replace("SEED", str(args.seed))
            script.write(current+"\n")
        script.write("\n")
        for b in proportions:
            current = T.replace(
                "PROP", str(b)
            ).replace(
                "elbo", "mse"
            ).replace(
                "CNN", "RNN"
            ).replace(
                "cnn", "rnn"
            ).replace("SEED", str(args.seed))+" -u"
            script.write(current+"\n")
        script.close()
        os.system("sh "+args.file)
