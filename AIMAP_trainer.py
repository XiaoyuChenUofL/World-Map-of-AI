import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from reco_utils.evaluation.python_evaluation import (rmse, mae, rsquared, exp_var, map_at_k, ndcg_at_k, precision_at_k,recall_at_k, get_top_k_items, merge_ranking_true_pred)
from sklearn.metrics.pairwise import cosine_similarity
from numpy import linalg as LA
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.tri as mtri
from sklearn.preprocessing import Normalizer, StandardScaler, MinMaxScaler
import itertools
from itertools import product
import matplotlib
import matplotlib.cm as cm
from pylab import * # For adjusting frame width only

from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score, mean_squared_error, roc_auc_score
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




def train(train_loader, datasets_X, pipelines_X, models, criterions, optimizer, alpha1, alpha2, alpha3,alpha4):
    autoencoder1, autoencoder2, decoder = models
    autoencoder1.train()
    autoencoder2.train()
    decoder.train()   
    train_nb_batch = len(train_loader)
    train_total_loss = 0.0
    train_total_auc = 0.0
    train_total_acc = 0.0
    ####################################
    for i, (minibatch_X, minibatch_y, minibatch_rate) in enumerate(train_loader):
        minibatch_X = minibatch_X.to(device)
        minibatch_y = minibatch_y.to(device).long()
        datasets_embed, mu_datasets, logvar_datasets = autoencoder1.enc(datasets_X)
        datasets_out = autoencoder1.dec(datasets_embed)
        pipelines_embed, mu_pipelines, logvar_pipelines = autoencoder2.enc(pipelines_X)
        pipelines_out = autoencoder2.dec(pipelines_embed)
        # print(datasets_embed.shape, pipelines_embed.shape, minibatch_X.shape)
        outputs = decoder(datasets_embed, pipelines_embed,minibatch_X)
        #############################################
        #how to define customized loss? 
        criterion1, criterion2 = criterions
        loss1 = criterion1(outputs, minibatch_y)
        loss2 = criterion2(pipelines_out, pipelines_X)
        loss3 = criterion2(datasets_out, datasets_X)
        KLD = (-0.5 * torch.sum(1 + logvar_datasets - mu_datasets.pow(2) - logvar_datasets.exp()) )+(-0.5 * torch.sum(1 + logvar_pipelines - mu_pipelines.pow(2) - logvar_pipelines.exp()) )
        loss = (alpha1*loss1 + alpha2*loss2+ alpha3*loss3 + alpha4*KLD)
#         loss = loss1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_batch_loss = loss.item()
        train_total_loss += train_batch_loss
#         outputs = F.softmax(outputs,dim=1)
        pred_batch_train = outputs.data.max(1, keepdim=True)[1]
        train_batch_acc = accuracy_score(minibatch_y,pred_batch_train)
        train_total_acc += train_batch_acc
        train_batch_auc = roc_auc_score(minibatch_y.detach().numpy(), pred_batch_train.detach().numpy() )
        train_total_auc += train_batch_auc
        if i == 0 :
            pred_train = pred_batch_train
        if i >0:
            pred_train = torch.cat([pred_train, pred_batch_train], dim=0)
    #############################################     
    train_loss = train_total_loss/train_nb_batch
    train_acc = train_total_acc/train_nb_batch
    train_auc = train_total_auc/train_nb_batch
#     scheduler.step()  
    return train_loss, train_acc, train_auc, pred_train





def test(test_loader, datasets_X, pipelines_X, models, criterions, optimizer, alpha1, alpha2, alpha3,alpha4):
    autoencoder1, autoencoder2,decoder = models 
    autoencoder1.eval()
    autoencoder2.eval()
    decoder.eval()
    test_nb_batch = len(test_loader)
    test_total_loss = 0.0
    test_total_auc = 0.0
    test_total_acc = 0.0
    ####################################
    for i, (minibatch_X, minibatch_y, minibatch_rate) in enumerate(test_loader):
        minibatch_X = minibatch_X.to(device)
        minibatch_y = minibatch_y.to(device).long()
        datasets_embed, mu_datasets, logvar_datasets = autoencoder1.enc(datasets_X)
        datasets_out = autoencoder1.dec(datasets_embed)
        pipelines_embed, mu_pipelines, logvar_pipelines = autoencoder2.enc(pipelines_X)
        pipelines_out = autoencoder2.dec(pipelines_embed)
        outputs = decoder(datasets_embed, pipelines_embed,minibatch_X)
        #############################################
        #how to define customized loss? 
        criterion1, criterion2 = criterions
        loss1 = criterion1(outputs, minibatch_y)
        loss2 = criterion2(pipelines_out, pipelines_X)
        loss3 = criterion2(datasets_out, datasets_X)
        KLD = (-0.5 * torch.sum(1 + logvar_datasets - mu_datasets.pow(2) - logvar_datasets.exp()) )+(-0.5 * torch.sum(1 + logvar_pipelines - mu_pipelines.pow(2) - logvar_pipelines.exp()) )
        loss = (alpha1*loss1 + alpha2*loss2+ alpha3*loss3 + alpha4*KLD)

        test_batch_loss = loss.item()
        test_total_loss += test_batch_loss
#         outputs = F.softmax(outputs,dim=1)
        pred_batch_test = outputs.data.max(1, keepdim=True)[1]
        test_batch_acc = accuracy_score(minibatch_y,pred_batch_test)
        test_total_acc += test_batch_acc
        test_batch_auc = roc_auc_score(minibatch_y.detach().numpy(), pred_batch_test.detach().numpy() )
        test_total_auc += test_batch_auc
        if i == 0 :
            pred_test = pred_batch_test
        if i >0:
            pred_test = torch.cat([pred_test, pred_batch_test], dim=0)
    #############################################     
    test_loss = test_total_loss/test_nb_batch
    test_acc = test_total_acc/test_nb_batch
    test_auc = test_total_auc/test_nb_batch    
    return test_loss, test_acc, test_auc, pred_test





def fit(EPOCHS, train_loader, test_loader, datasets_X, pipelines_X, models, criterions, optimizer, alpha1, alpha2, alpha3, alpha4):
    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []
    train_aucs = []
    test_aucs = []

    for epoch in range(EPOCHS):
        train_loss, train_acc, train_auc, pred_train = train(train_loader, datasets_X, pipelines_X, models, criterions, optimizer, alpha1, alpha2, alpha3,alpha4)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        train_aucs.append(train_auc)
        print("epoch:",epoch,", | train loss:", train_loss, ", | train acc:", train_acc, ", | train auc:", train_auc)
        
        test_loss, test_acc, test_auc, pred_test = test(test_loader, datasets_X, pipelines_X, models, criterions, optimizer, alpha1, alpha2, alpha3,alpha4)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        test_aucs.append(test_auc)
        print("epoch:",epoch,", | test loss:", test_loss,", | test acc:", test_acc, ", | test auc:", test_auc)

        #save the model if interested
        # if (epoch+1) % 5 == 0:
        # PATH = 'saved_models/VAE-NCF/encoder1/10222021/v2_encoder1_VAE-NCF_epoch%d_1lambda1_2lambda2_2lambda3_e-5lambda4.pkl'%(epoch+1)
        # torch.save(autoencoder1, PATH)
        # PATH = 'saved_models/VAE-NCF/encoder2/10222021/v2_ecoder2_VAE-NCF_epoch%d_1lambda1_2lambda2_2lambda3_e-5lambda4.pkl'%(epoch+1)
        # torch.save(autoencoder2, PATH)
        # PATH = 'saved_models/VAE-NCF/decoder/10222021/v2_decoder_VAE-NCF_epoch%d_1lambda1_2lambda2_2lambda3_e-5lambda4.pkl'%(epoch+1)
        # torch.save(decoder, PATH)

        #save the loss and accs if inteested
