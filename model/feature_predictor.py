import torch
import torch.nn as nn
from torch.autograd import Variable
import sys
import torch
import glob
import unicodedata
import string
import logging

all_letters = string.ascii_letters + " .;'-"
n_letters = len(all_letters)
all_categories = ['power', 'price', 'exterior', 'configuration']

logger = logging.getLogger("darwin")
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

rnn = RNN(input_size=57, hidden_size=128, output_size=4)

try:
    rnn.load_state_dict(torch.load('./feature_predictor.pt')) # load in same dir
except FileNotFoundError: 
    rnn.load_state_dict(torch.load('../model/feature_predictor.pt')) # imported by other scripts
    

# ------------------------------------------
# The below code credits to https://github.com/spro/practical-pytorch/blob/master/char-rnn-classification/data.py        

def findFiles(path): return glob.glob(path)

# Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

# Read a file and split into lines
def readLines(filename):
    lines = open(filename).read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

n_categories = len(all_categories)

# Find letter index from all_letters, e.g. "a" = 0
def letterToIndex(letter):
    return all_letters.find(letter)

# Just for demonstration, turn a letter into a <1 x n_letters> Tensor
def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor

# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor
    
def evaluate(line_tensor):
    hidden = rnn.initHidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    return output

def predict(line, n_predictions=3):
        output = evaluate(Variable(lineToTensor(line)))
        # Get top N categories
        topv, topi = output.data.topk(n_predictions, 1, True)
        predictions = []

        for i in range(n_predictions):
            value = topv[0][i]
            category_index = topi[0][i]
            # print('(%.2f) %s' % (value, all_categories[category_index]))
            predictions.append([value.data.tolist(), all_categories[category_index]])
        return predictions

# ------------------------------------------
# Ends credit


if __name__ == '__main__':
    predict(sys.argv[1])