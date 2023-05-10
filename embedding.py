# import libraries
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
import re
import matplotlib.pyplot as plt
import math
import warnings
warnings.filterwarnings('ignore')
import torch
from transformers import BertTokenizer, BertModel

# read in the training data
train = pd.read_csv(r'C:\Users\不知道叫什么\Desktop\HKU sme2\ARIN7102 Applied data mining and text analytics\project/Train_rev1.csv')

# define functions for data cleaning
def convert_utf8(s):
    return str(s)

def remove_urls(s):
    s = re.sub('[^\s]*.com[^\s]*', "", s)
    s = re.sub('[^\s]*www.[^\s]*', "", s)
    s = re.sub('[^\s]*.co.uk[^\s]*', "", s)
    return s

def remove_star_words(s):
    return re.sub('[^\s]*[\*]+[^\s]*', "", s)

def remove_nums(s):
    return re.sub('[^\s]*[0-9]+[^\s]*', "", s)

from string import punctuation

def remove_punctuation(s):
    global punctuation
    for p in punctuation:
        s = s.replace(p, '')
    return s

# apply the cleaning functions to the FullDescription column in the training data
train['Clean_Full_Descriptions'] = train['FullDescription'].map(remove_urls)
train['Clean_Full_Descriptions'] = train['Clean_Full_Descriptions'].map(remove_star_words)
train['Clean_Full_Descriptions'] = train['Clean_Full_Descriptions'].map(remove_nums)
train['Clean_Full_Descriptions'] = train['Clean_Full_Descriptions'].map(remove_punctuation)
train['Clean_Full_Descriptions'] = train['Clean_Full_Descriptions'].map(lambda x: x.lower())

# extract columns of interest
Description = train['Clean_Full_Descriptions']
Title = train['Title']
Category = train['Category']

# initialize the BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# define a function to get the BERT embeddings for a given text input
def embedding(text):
    # encode the text using the BERT tokenizer
    input_ids = tokenizer.encode(text, add_special_tokens=True)

    # convert the tokenized sequence to a PyTorch tensor
    input_tensor = torch.tensor([input_ids])

    # generate the BERT embeddings
    with torch.no_grad():
        outputs = model(input_tensor)
        embeddings = outputs[0][0][1:-1]  # ignore the special tokens and take the embeddings of the original text only

    # convert the embeddings to a 100-dimensional vector
    embeddings_100d = torch.mean(embeddings, dim=0)[:100].numpy()
    return embeddings_100d

# loop through the Description column in the training data and get the BERT embeddings for each text input
result = []
for i, text in enumerate(Description):
    vector = embedding(text)
    result.append(vector)

# save the resulting embedding matrix as a numpy file
result = np.array(result)
np.save('job_matrix',result)
