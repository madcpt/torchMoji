"""Finetuning example.

Trains the torchMoji model on the SS-Youtube dataset, using the 'last'
finetuning method and the accuracy metric.

The 'last' method does the following:
0) Load all weights except for the softmax layer. Do not add tokens to the
   vocabulary and do not extend the embedding layer.
1) Freeze all layers except for the softmax layer.
2) Train.
"""

from __future__ import print_function
import example_helper
import json
import torch
import numpy as np
from torchmoji.model_def import torchmoji_transfer
from torchmoji.global_variables import PRETRAINED_PATH, VOCAB_PATH, ROOT_PATH
from torchmoji.finetuning import (
    load_benchmark,
    finetune)

import argparse


# add args
parser = argparse.ArgumentParser(description='Fine-Tune')
parser.add_argument('-ds','--dataset', help='dataset', required=True, default="SS-Youtube")
parser.add_argument('-m','--method', help='method', required=True, default="last")
parser.add_argument('-n','--nb_classes', help='nb_classes', required=True, default=2)

args = vars(parser.parse_args())
print(args)

torch.autograd.set_detect_anomaly(True)

DATASET_PATH = '{}/data/{}/raw.pickle'.format(ROOT_PATH, args['dataset'])

nb_classes = int(args['nb_classes'])

with open(VOCAB_PATH, 'r') as f:
    vocab = json.load(f)

# Load dataset.
data = load_benchmark(DATASET_PATH, vocab)

device = torch.device('cuda:0')

# Set up model and finetune
model = torchmoji_transfer(nb_classes, PRETRAINED_PATH).to(device)
print(model)
# print(torch.tensor(data['texts'][0], dtype=torch.int32))
# print(torch.tensor(data['texts'][0].astype(np.int32)))
model, acc = finetune(model, data['texts'], data['labels'], nb_classes, data['batch_size'], method=args['method'], device=device, dataset=args['dataset'])
print('Acc: {}'.format(acc))

with open('{}/log/{}-{}-best.log'.format(ROOT_PATH, args['dataset'], args['method']), 'w') as f:
    f.write('test: %.4f\n' % acc)