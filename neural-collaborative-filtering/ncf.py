import torch
import torch.nn as nn
import torch.nn.functional as F

class NCF(nn.Module):
    def __init__(
        self,
        num_users,
        num_items,
        factors=8,
        layers=[64, 32, 16, 8],
        dropout=0
    ):
        super(NCF, self).__init__()
        
        # Embedding layers
        self.mf_user_embedding = nn.Embedding(num_users, factors)
        self.mf_item_embedding = nn.Embedding(num_items, factors)
        
        self.mlp_user_embedding = nn.Embedding(num_users, int(layers[0]/2))
        self.mlp_item_embedding = nn.Embedding(num_items, int(layers[0]/2))

        # MLP layers
        self.dropout = dropout
        self.mlp_layers = nn.ModuleList()
        for idx, (in_dim, out_dim) in enumerate(zip(layers[:-1], layers[1:])):
            self.mlp_layers.append(nn.Linear(in_dim, out_dim))
        
        self.prediction = nn.Linear(factors + layers[-1], 1)

    def forward(self, user, item):
        mf_user_embed = self.mf_user_embedding(user)
        mf_item_embed = self.mf_item_embedding(item)
        mf_vector = torch.mul(mf_user_embed, mf_item_embed)
        
        mlp_user_embed = self.mlp_user_embedding(user)
        mlp_item_embed = self.mlp_item_embedding(item)
        mlp_vector = torch.cat([mlp_user_embed, mlp_item_embed], dim=-1)
        
        for layer in self.mlp_layers:
            mlp_vector = F.relu(layer(mlp_vector))
            if self.dropout > 0:
                mlp_vector = F.dropout(mlp_vector, p=self.dropout, training=self.training)
        
        vector = torch.cat([mf_vector, mlp_vector], dim=-1)
        logits = self.prediction(vector)
        output = torch.sigmoid(logits)
        
        return output.view(-1)
