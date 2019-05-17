import text_processing
import torch
import torch.nn.functional as F
from imp import reload
import models
reload(models)
from text_processing import read_split_file_lyrics, BatchWrapper, generate_iterators_lyrics
from functions import *
import numpy as np
import pandas as pd
from models import BiLSTMSourceNet

print("Reading in the cleaned lyrics data.")

#Read and split data
train_path, val_path = read_split_file_lyrics("lyrics_cleaned.csv.gz","")

print("Creating iterators.")
#Create iterators
torch.cuda.empty_cache() 
batch_size = 10
train_iter, val_iter, TEXT = generate_iterators_lyrics(train_path, val_path, 
                                                           batch_size = batch_size, device="cuda")


print("Running randomized hyper-parameter search.")

dropout_min = 0.2
dropout_max = 0.8

learning_rate_min = 0.0001
learning_rate_max = 0.01

hidden_dim_min = 50
hidden_dim_max = 300

v_dim_min = 5
v_dim_max = 30
label_size = 2
embedding_dim = 300

dropouts = []
learning_rates = []
hidden_dims = []
v_dims = []
dev_accuracies = []

best_dev_acc = 0
best_hidden_dim = 0
best_v_dim = 0
best_learning_rate = 0
best_dropout = 0
best_m_source = None
src_attention_songs_final = None

for i in range(50):
    print("Iteration: {}".format(i+1))
    #Create wrappers for iterators
    train_batch = BatchWrapper(train_iter, "lyrics", ["pos", "neg"])
    valid_batch = BatchWrapper(val_iter, "lyrics", ["pos", "neg"])
    
    dropout = np.random.uniform(dropout_min, dropout_max)
    learning_rate = 10**np.random.uniform(np.log10(learning_rate_min), np.log10(learning_rate_max))
    hidden_dim = np.random.randint(hidden_dim_min, hidden_dim_max)
    v_dim = np.random.randint(v_dim_min, v_dim_max)
    
    m_source = BiLSTMSourceNet(vocab_size = len(TEXT.vocab), embedding_dim = embedding_dim, 
                               hidden_dim = hidden_dim, label_size=label_size, v_dim = v_dim, 
                               pretrained_vec=TEXT.vocab.vectors, use_gpu = True, dropout = dropout)
    m_source.to("cuda")
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, m_source.parameters()), learning_rate)
    
    dev_acc, src_attention, H_src = training_loop_DATN(model=m_source, training_iter=train_batch, 
                                                          dev_iter=valid_batch, loss_=F.kl_div, optim=opt, num_epochs=10, 
                                                          source_flag = True, batch_size = batch_size, verbose = False)

    dropouts.append(dropout)
    learning_rates.append(learning_rate)
    hidden_dims.append(hidden_dim)
    v_dims.append(v_dim)
    dev_accuracies.append(dev_acc)
    
    try:
        check = dev_accuracies[-1] > dev_accuracies[-2]
    except:
        check = dev_accuracies[-1] > 0
        
    if check:
        
        best_dev_acc = dev_acc
        best_learning_rate = learning_rates[-1]
        best_hidden_dim = hidden_dims[-1]
        best_v_dim  = v_dims[-1]
        best_dropout = dropouts[-1]
        best_m_source = m_source
        src_attention_songs_final = src_attention
    
    df = pd.DataFrame({"droprate": dropouts, "learning_rates": learning_rates, "hidden_dim": hidden_dims, "v_dim": v_dims, "dev_accuracy": dev_accuracies})
    df.to_csv("lyrics_results_source_dec2-6_YD.csv")

df = pd.DataFrame({"droprate": dropouts, "learning_rates": learning_rates, "hidden_dim": hidden_dims, "v_dim": v_dims, "dev_accuracy": dev_accuracies})

df_2 = pd.DataFrame({"droprate": best_dropout, "learning_rates": best_learning_rate, "hidden_dim": best_hidden_dim, "v_dim": best_v_dim, "dev_accuracy": best_dev_acc}, index = [0])

df_2.to_csv("lyrics_results_source_best_dec2-6_YD.csv")
# torch.save(best_m_source.state_dict(), "/SourceNetModelDict")
