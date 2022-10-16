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
from AIMAP_networks import Encoder, Decoder, AE, NCF, MultiLayerPerceptron
from AIMAP_trainer import train, test, fit




#load the metadata and pipeline embedding vectors
input_dim = 50
latent_dim = 2
scaler = StandardScaler()
encoder1 = Encoder(input_dim , latent_dim)
encoder2 = Encoder(input_dim, latent_dim)
decoder = NCF(latent_dim, mlp_dims=(32,16,8,4), dropout=0.5)
datasets_dense_df = pd.read_csv('datasets_metadata_vec_v2.csv')
datasets_dense = datasets_dense_df.values
pipelines_glove_df = pd.read_csv('pipelines_embedding_vec_v2.csv')
pipelines_dense = pipelines_glove_df.values

print(datasets_dense.shape,pipelines_dense.shape)
scaler.fit(datasets_dense)
datasets_dense =scaler.transform(datasets_dense)
scaler.fit(pipelines_dense)
pipelines_dense =scaler.transform(pipelines_dense)
#crate torch dataset and data loader for meta data and pipeline embeddings
datasets_dense_torch = torch.from_numpy(datasets_dense)
pipelines_dense_torch = torch.from_numpy(pipelines_dense)
datasets_data = TensorDataset(datasets_dense_torch)
pipelines_data = TensorDataset(pipelines_dense_torch)
datasets_loader = DataLoader(datasets_data, batch_size= len(datasets_data), num_workers=0)
pipelines_loader = DataLoader(pipelines_data, batch_size= len(pipelines_data), num_workers=0)
__ , datasets_X = next(enumerate(datasets_loader))
__ , pipelines_X = next(enumerate(pipelines_loader))
datasets_X = datasets_X[0].float().to(device)
pipelines_X = pipelines_X[0].float().to(device)




#load the dataset-pipeline interaction performances:
dp_interaction_df = pd.read_csv('Dataset_Pipeline_interaction.csv')
dp_interaction = dp_interaction_df.values
#torch dataset and data loader
train_length = int(len(dp_interaction[:,0]) * 0.8)
test_length = len(dp_interaction[:,0]) - train_length
features = torch.from_numpy(dp_interaction[:,:2])
scores = torch.from_numpy(dp_interaction[:,2])
bin_targets = torch.from_numpy(dp_interaction[:,3])
all_data = TensorDataset(features, bin_targets, scores)
train_data, test_data = torch.utils.data.random_split(all_data, (train_length, test_length))
train_batch_size = 64
test_batch_size = 64
train_loader = DataLoader(train_data, batch_size=train_batch_size, num_workers=0)
test_loader = DataLoader(test_data, batch_size=test_batch_size, num_workers=0)




#model configurations:
autoencoder1 = AE(input_dim=50 , latent_dim=2)
autoencoder1 = autoencoder1.to(device)
autoencoder2 = AE(input_dim=50 , latent_dim=2)
autoencoder2 = autoencoder2.to(device)
decoder = NCF(latent_dim=2, mlp_dims=(16,8,4), dropout=0.1)
decoder = decoder.to(device)

criterion1 = nn.CrossEntropyLoss()
criterion2 = nn.MSELoss()
params = list(decoder.parameters()) + list(autoencoder1.parameters()) + list(autoencoder2.parameters())
optimizer = optim.Adam(params, lr=0.01, amsgrad=False)
# scheduler = MultiStepLR(optimizer, milestones=[5,10,15,20,30,40,45], gamma=0.7)


#fit the model
EPOCHS = 20
models = (autoencoder1, autoencoder2, decoder)
criterions = (criterion1, criterion2)
alpha1 = 1
alpha2 = alpha3 = 2
alpha4 = 0.00001
fit(EPOCHS, train_loader, test_loader, datasets_X, pipelines_X, models, criterions, optimizer, alpha1, alpha2, alpha3, alpha4)


#load a pre-trained model:
#load models:
# PATH = 'AIMAP/saved_models/encoder1_VAE-NCF_epoch15_1lambda1_10lambda2_10lambda3_e-5lambda4.pkl'
# autoencoder1 = torch.load(PATH).to(device)
# PATH = 'AIMAP/saved_models/ecoder2_VAE-NCF_epoch15_1lambda1_10lambda2_10lambda3_e-5lambda4.pkl'
# autoencoder2 = torch.load(PATH).to(device)
# PATH = 'AIMAP/saved_models/decoder_VAE-NCF_epoch15_1lambda1_10lambda2_10lambda3_e-5lambda4.pkl'
# decoder = torch.load(PATH).to(device)


# #visualize latents:
# datasets_embed, __, __ = autoencoder1.enc.forward(datasets_dense_torch.float().to(device))
# datasets_embed = datasets_embed.cpu().detach().numpy()

