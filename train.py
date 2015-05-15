#!/usr/bin/env python

import argparse
import dlm.utils as U
import dlm.io.logging as L

###############
## Arguments
#

parser = argparse.ArgumentParser()
parser.add_argument("-tr", "--train-mmap", dest="trainset", required=True, help="The memory-mapped training file")
parser.add_argument("-tu", "--tune-mmap", dest="devset", required=True, help="The memory-mapped development (tune) file")
parser.add_argument("-ts", "--test-mmap", dest="testset", help="The memory-mapped evaluation (test) file")
parser.add_argument("-M", "--model", dest="model_path", required=True, help="The output model file")
parser.add_argument("-D", "--device", dest="device", default="gpu", help="The computing device (cpu or gpu)")
parser.add_argument("-E", "--emb-dim", dest="emb_dim", default=50, type=int, help="Word embeddings dimension")
parser.add_argument("-H", "--hidden-units", dest="num_hidden", default="512,0", help="A comma seperated list for the number of units in each hidden layer")
parser.add_argument("-A", "--activation", dest="activation_name", default="tanh", help="Activation function (tanh|hardtanh|sigmoid|fastsigmoid|hardsigmoid|softplus|relu|cappedrelu)")
parser.add_argument("-a", "--training-algorithm", dest="algorithm", default="sgd", help="The training algorithm (only sgd is supported for now)")
parser.add_argument("-b", "--batch-size", dest="batchsize", default=128, type=int, help="Minibatch size for training")
parser.add_argument("-l", "--learning-rate", dest="learning_rate", default=0.01, type=float, help="Learning rate")
parser.add_argument("-e", "--num-epochs", dest="num_epochs", default=50, type=int, help="Number of iterations (epochs)")
parser.add_argument("-c", "--self-norm-coef", dest="alpha", default=0, type=float, help="Self normalization coefficient (alpha)")
parser.add_argument("-L1", "--L1-regularizer", dest="L1_reg", default=0, type=float, help="L1 regularization coefficient")
parser.add_argument("-L2", "--L2-regularizer", dest="L2_reg", default=0, type=float, help="L2 regularization coefficient")
parser.add_argument("-log", "--log-file", dest="log_file", default='primelm.train.log', help="The output log file")
args = parser.parse_args()

L.set_file_path(args.log_file)

U.set_theano_device(args.device)

import dlm.trainer
from dlm.io.mmapReader import MemMapReader
from dlm.criterions.likelihood import NegLogLikelihood
from dlm.models.ltmlp import MLP

#########################
## Loading datasets
#

trainset = MemMapReader(args.trainset, batch_size=args.batchsize)
devset = MemMapReader(args.devset)
testset = None
if args.testset:
	testset = MemMapReader(args.testset)

#########################
## Creating model
#

L.info('Building the model')

args.vocab_size = trainset.get_vocab_size()
args.ngram_size = trainset.get_ngram_size()
args.num_classes = trainset.get_num_classes()

classifier = MLP(args)

#########################
## Training criterion
#

criterion = NegLogLikelihood(classifier, args)

#########################
## Training
#

dlm.trainer.train(classifier, criterion, args, trainset, devset, testset)





