#!/bin/sh

python reproduce_DeepCRISPR.py -o ../models/DeepCRISPR/hek293t/cnn_elbo_v -l hek293t -m CNN 
python reproduce_DeepCRISPR.py -o ../models/DeepCRISPR/hek293t/cnn_mse_v -l hek293t -m CNN -u 
python reproduce_DeepCRISPR.py -o ../models/DeepCRISPR/hek293t/rnn_elbo_v -l hek293t -m RNN 
python reproduce_DeepCRISPR.py -o ../models/DeepCRISPR/hek293t/rnn_mse_v -l hek293t -m RNN -u 


python reproduce_DeepCRISPR.py -o ../models/DeepCRISPR/hela/cnn_elbo_v -l hela -m CNN 
python reproduce_DeepCRISPR.py -o ../models/DeepCRISPR/hela/cnn_mse_v -l hela -m CNN -u 
python reproduce_DeepCRISPR.py -o ../models/DeepCRISPR/hela/rnn_elbo_v -l hela -m RNN 
python reproduce_DeepCRISPR.py -o ../models/DeepCRISPR/hela/rnn_mse_v -l hela -m RNN -u 


python reproduce_DeepCRISPR.py -o ../models/DeepCRISPR/hct116/cnn_elbo_v -l hct116 -m CNN 
python reproduce_DeepCRISPR.py -o ../models/DeepCRISPR/hct116/cnn_mse_v -l hct116 -m CNN -u 
python reproduce_DeepCRISPR.py -o ../models/DeepCRISPR/hct116/rnn_elbo_v -l hct116 -m RNN 
python reproduce_DeepCRISPR.py -o ../models/DeepCRISPR/hct116/rnn_mse_v -l hct116 -m RNN -u 


python reproduce_DeepCRISPR.py -o ../models/DeepCRISPR/hl60/cnn_elbo_v -l hl60 -m CNN 
python reproduce_DeepCRISPR.py -o ../models/DeepCRISPR/hl60/cnn_mse_v -l hl60 -m CNN -u 
python reproduce_DeepCRISPR.py -o ../models/DeepCRISPR/hl60/rnn_elbo_v -l hl60 -m RNN 
python reproduce_DeepCRISPR.py -o ../models/DeepCRISPR/hl60/rnn_mse_v -l hl60 -m RNN -u 