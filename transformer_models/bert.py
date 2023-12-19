import math
import re
from random import *
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

text = (
       'Hello, how are you? I am Romeo.n'
       'Hello, Romeo My name is Juliet. Nice to meet you.n'
       'Nice meet you too. How are you today?n'
       'Great. My baseball team won the competition.n'
       'Oh Congratulations, Julietn'
       'Thanks you Romeo'
   )

sentences = re.sub("[.,!?-]", '', text.lower()).split('n')

word_list = list(set("".join(sentences).split()))

word_dict =  {'[PAD]':0, '[CLS]':1, '[SEP]':2, '[MASK]':3}
for i, w in enumerate(word_list):
    word_dict[w] = i+4
    number_dict = {i+4}
    number_dict = {i: w for i,w in enumerate(word_dict)}
    vocab_size = len(word_dict)

