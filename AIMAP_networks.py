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
from sklearn.metrics.pairwise import cosine_similarity
from numpy import linalg as LA
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.tri as mtri
from sklearn.preprocessing import Normalizer, StandardScaler, MinMaxScaler
import itertools
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score, mean_squared_error, roc_auc_score
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




class MultiLayerPerceptron(torch.nn.Module):
    """
    Class to instantiate a Multilayer Perceptron model
    """
    def __init__(self, input_dim, embed_dims, dropout, output_layer=True):
        super().__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.Tanh())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        return self.mlp(x)



class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.fc1 = nn.Linear(self.input_dim, 32)
        self.dropout = nn.Dropout(0.3)
        self.bn = nn.BatchNorm2d(32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 8)
        self.fc4 = nn.Linear(8, 4)
        self.fc51 = nn.Linear(4, self.latent_dim)
        self.fc52 = nn.Linear(4, self.latent_dim)
   
    def forward(self, x):
        # define your feedforward pass
        h1 = torch.tanh(self.fc1(x))
        h2 = self.fc2(h1)
        h3 = self.fc3(h2) 
        h4 = self.fc4(h3)
        mean = self.fc51(h4)
        logvar = self.fc52(h4) 
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        out = mean + eps*std
        return out, mean, logvar



class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Decoder, self).__init__()
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.fc6 = nn.Linear(self.latent_dim, 4)
        self.fc7 = nn.Linear(4, 8)
        self.fc8 = nn.Linear(8, 16)
        self.fc9 = nn.Linear(16, 32)
        self.fc10 = nn.Linear(32, self.output_dim)

    def forward(self, x):
        # define your feedforward pass
        h5 = torch.tanh(self.fc6(x))
#         h5 = self.dropout(h5)
        h6 = self.fc7(h5)
#         h6 = self.dropout(h6)
        h7 = self.fc8(h6)
#         h7 = self.bn(h7)
#         h7 = self.dropout(h7)
        h8 = self.fc9(h7)
        recontructed = self.fc10(h8)
        return recontructed



class AE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(AE, self).__init__()
        self.input_dim = input_dim
        self.out_dim = input_dim
        self.latent_dim = latent_dim
        self.enc = Encoder(self.input_dim, self.latent_dim)
        self.dec = Decoder(self.latent_dim, self.out_dim)
    def forward(self, x):
        # define your feedforward pass
        latent = self.enc.forward(x.view(-1, self.input_dim))
        out = self.decode(latent)
        return out
    def decode(self, z):
        # given random noise z, generate airfoils
        return self.dec.forward(z)



class NCF(nn.Module):
    def __init__(self, latent_dim, mlp_dims, dropout):
        super().__init__()
        self.latent_dim = latent_dim
        self.mlp = MultiLayerPerceptron(2*self.latent_dim, mlp_dims, dropout, output_layer=False)
        self.fc = torch.nn.Linear(mlp_dims[-1] + latent_dim, 2)

    def forward(self, datasets_embed, pipelines_embed,x):
        product_dataset_pipeline = torch.zeros((len(x), self.latent_dim))
        concat_dataset_pipeline = torch.zeros((len(x), 2*self.latent_dim))
        for s in range(len(x)):
            i = x[s,:][0].item()
            j = x[s,:][1].item()
            i = int(i)
            j = int(j)
            product_dataset_pipeline[s,:] = datasets_embed[i,:]*pipelines_embed[j,:]
            concat_dataset_pipeline[s,:] = torch.cat([datasets_embed[i,:], pipelines_embed[j,:]], dim=0)
        mlp = self.mlp(concat_dataset_pipeline)
        gmf = product_dataset_pipeline
#         gmf = torch.sigmoid(gmf)
        x = torch.cat([gmf, mlp], dim=1)
        x = self.fc(x).squeeze(1)
        return x
