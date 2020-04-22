#!/bin/sh


python reproduce_geCRISPR.py -m CNN -o ../models/geCRISPR/cnn_mse -u
python reproduce_geCRISPR.py -m RNN -o ../models/geCRISPR/rnn_mse -u
python reproduce_geCRISPR.py -m CNN -o ../models/geCRISPR/cnn_elbo 
python reproduce_geCRISPR.py -m RNN -o ../models/geCRISPR/rnn_elbo 
python reproduce_DeepHF.py -o ../models/DeepHF/Wt/rnn_mse -d Wt -m RNN -u
python reproduce_DeepHF.py -o ../models/DeepHF/Wt/cnn_mse -d Wt -m CNN -u
python reproduce_DeepHF.py -o ../models/DeepHF/eSpCas9/rnn_mse -d eSpCas9 -m RNN -u
python reproduce_DeepHF.py -o ../models/DeepHF/eSpCas9/cnn_mse -d eSpCas9 -m CNN -u
python reproduce_DeepHF.py -o ../models/DeepHF/SpCas9HF1/rnn_mse -d SpCas9HF1 -m RNN -u
python reproduce_DeepHF.py -o ../models/DeepHF/SpCas9HF1/cnn_mse -d SpCas9HF1 -m CNN -u
python reproduce_DeepHF.py -o ../models/DeepHF/Wt/rnn_elbo -d Wt -m RNN
python reproduce_DeepHF.py -o ../models/DeepHF/Wt/cnn_elbo -d Wt -m CNN
python reproduce_DeepHF.py -o ../models/DeepHF/eSpCas9/rnn_elbo -d eSpCas9 -m RNN
python reproduce_DeepHF.py -o ../models/DeepHF/eSpCas9/cnn_elbo -d eSpCas9 -m CNN
python reproduce_DeepHF.py -o ../models/DeepHF/SpCas9HF1/rnn_elbo -d SpCas9HF1 -m RNN
python reproduce_DeepHF.py -o ../models/DeepHF/SpCas9HF1/cnn_elbo -d SpCas9HF1 -m CNN