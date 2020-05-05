import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2, z_dim):
        super(Encoder, self).__init__()
        self.dense1 = nn.Linear(in_features=x_dim, out_features=h_dim1, bias=True)
        self.batch_norm1 = nn.BatchNorm1d(h_dim1)
        self.dense2 = nn.Linear(in_features=h_dim1, out_features=h_dim2, bias=True)
        self.batch_norm2 = nn.BatchNorm1d(h_dim2)
        self.dense3 = nn.Linear(in_features=h_dim2, out_features=z_dim, bias=True)
        self.batch_norm3 = nn.BatchNorm1d(z_dim)
        
    
    def forward(self, x):
        x = F.relu(self.batch_norm1(self.dense1(x)))
        x = F.relu(self.batch_norm2(self.dense2(x)))
        x = F.relu(self.batch_norm3(self.dense3(x)))
        return x

class Decoder(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2, z_dim):
        super(Decoder, self).__init__()
        self.dense1 = nn.Linear(in_features=z_dim, out_features=h_dim2, bias=True)
        self.batch_norm1 = nn.BatchNorm1d(h_dim2)
        self.dense2 = nn.Linear(in_features=h_dim2, out_features=h_dim1, bias=True)
        self.batch_norm2 = nn.BatchNorm1d(h_dim1)
        self.dense3 = nn.Linear(in_features=h_dim1, out_features=x_dim, bias=True)
        self.batch_norm3 = nn.BatchNorm1d(x_dim)
        
    
    def forward(self, x):
        x = F.relu(self.batch_norm1(self.dense1(x)))
        x = F.relu(self.batch_norm2(self.dense2(x)))
        x = F.relu(self.batch_norm3(self.dense3(x)))
        return x  

class DenseAutoencoder(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2, z_dim):
        super(DenseAutoencoder, self).__init__()

        self.encoder = Encoder(x_dim, h_dim1, h_dim2, z_dim)
        self.decoder = Decoder(x_dim, h_dim1, h_dim2, z_dim)

    def forward(self, x):
        z = self.encoder(x)
        x = self.decoder(z)
        return x