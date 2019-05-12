#Structured code made originally by Micaela
import numpy as np
import pandas as pd
import spacy
import torch
from torchtext.data import Field, TabularDataset, Iterator, BucketIterator
from torchtext import vocab
from sklearn.model_selection import train_test_split
import sys
import csv

csv.field_size_limit(sys.maxsize)

spacy_en = spacy.load('en')
#Structured code made originally by Micaela
def read_split_file(data_source, tar_path):
    df = pd.read_csv(data_source)
    for e in df.columns[1:]:
        df[e] = df[e].astype(int)
    #split into train/val/test
    df_train, df_test = train_test_split(df, test_size = 0.2)
    df_train, df_val = train_test_split(df_train, test_size = 0.2)

    #write out to train/val/test csv
    train_path, val_path, test_path = [ tar_path  + s for s in ["poems_train.csv", "poems_val.csv", "poems_test.csv"] ]

    df_train.to_csv(train_path,index=False) #80% of the original 80%
    df_val.to_csv(val_path,index=False) #20% of the original 80%
    df_test.to_csv(test_path,index=False) #20% of original data

    return train_path, val_path, test_path

def read_split_file_lyrics(data_source, tar_path):
    df = pd.read_csv(data_source)
    #split into train/val/test
    #df = df[:1000]
    df["pos"] =  (df["label"] == 1)*1
    df["neg"] =  (df["label"] == -1)*1
    df = df.drop("label", axis=1)
    df_train, df_val = train_test_split(df, test_size = 0.2)
    
    #write out to train/val/test csv
    train_path, val_path = [ tar_path  + s for s in ["lyrics_train.csv", "lyrics_val.csv"] ]

    df_train.to_csv(train_path,index=False)
    df_val.to_csv(val_path,index=False) 

    return train_path, val_path

def tokenizer(text): 
    return [tok.text for tok in spacy_en.tokenizer(text)]



class BatchWrapper:
    '''
    NOTE: 
    BucketIterator returns a Batch object instead of text index and labels. 
    Also, the Batch object is not iterable like pytorch Dataloader
    This is a wrapper to extract the text and labels + make the Batch iterable
    '''
    def __init__(self, dl, x_field, y_field):
        self.dl, self.x_field, self.y_field = dl, x_field, y_field
    def __len__(self):
        return len(self.dl)
    
    def __iter__(self):
        for batch in self.dl:
            X = getattr(batch, self.x_field)
            # concatenate y into a single tensor
            y = torch.cat([getattr(batch, feat).unsqueeze(1) for feat in self.y_field], dim=1).float()
            yield (X,y)

class BatchWrapperLyrics:
    '''
    NOTE: 
    BucketIterator returns a Batch object instead of text index and labels. 
    Also, the Batch object is not iterable like pytorch Dataloader
    This is a wrapper to extract the text and labels + make the Batch iterable
    '''
    def __init__(self, dl, x_field, y_field):
        self.dl, self.x_field, self.y_field = dl, x_field, y_field
    def __len__(self):
        return len(self.dl)
    
    def __iter__(self):
        for batch in self.dl:
            X = getattr(batch, self.x_field)
            # concatenate y into a single tensor
            y = torch.cat([getattr(batch, feat).unsqueeze(1) for feat in self.y_field], dim=1).float()
            yield (X,y)

def generate_iterators(train_path, val_path, test_path, batch_size, device="cpu"):
    TEXT = Field(sequential=True, tokenize=tokenizer, lower=True)
    LABEL = Field(sequential=False, use_vocab=False)

    datafields = [("text", TEXT), ("anger", LABEL),
                    ("anticipation", LABEL),("fear", LABEL),
                    ("joy", LABEL),("love", LABEL),
                    ("optimism", LABEL),('pess', LABEL),('sad', LABEL)]
    train, val, test = TabularDataset.splits(
                path='',
                train=train_path, validation=val_path,
                test=test_path,format='csv',
                skip_header=True,fields=datafields)
    TEXT.build_vocab(train, vectors='glove.6B.300d')

    train_iter = BucketIterator(train, batch_size=batch_size, device=device, 
                                sort_key = lambda x: len(x.text), 
                                sort_within_batch=False, repeat=False)
    val_iter = BucketIterator(val, batch_size=1, device=device, 
                              sort_key = lambda x: len(x.text), 
                                sort_within_batch=False, repeat=False)
    #val_iter = Iterator(val, batch_size=1, device=device, 
    #                         sort_key = lambda x: len(x.text), 
    #                          sort_within_batch=False, repeat=False, train=False)

    test_iter = Iterator(test, batch_size=1, device=device, 
                        sort=False, sort_within_batch=False, repeat=True)

    return train_iter, val_iter, test_iter, TEXT

def generate_iterators_lyrics(train_path, val_path, batch_size, device="cpu"):
    TEXT = Field(sequential=True, tokenize=tokenizer, lower=True)
    LABEL = Field(sequential=False, use_vocab=False)

    datafields = [("lyrics", TEXT), ("pos", LABEL), ("neg", LABEL)]
    train, val = TabularDataset.splits(
                path='',
                train=train_path, validation=val_path,format='csv',
                skip_header=True,fields=datafields)
    TEXT.build_vocab(train, vectors='glove.6B.300d')

    train_iter = BucketIterator(train, batch_size=batch_size, device=device, 
                                sort_key = lambda x: len(x.text), 
                                sort_within_batch=False, repeat=False)
    val_iter = BucketIterator(val, batch_size=batch_size, device=device, 
                              sort_key = lambda x: len(x.text), 
                                sort_within_batch=False, repeat=False)


    return train_iter, val_iter, TEXT