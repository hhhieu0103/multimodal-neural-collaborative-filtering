import torch
import torch.nn as nn
import torch.nn.functional as F

class NCF(nn.Module):
    def __init__(
        self,
        num_users,
        num_items,
        factors=8,
        mlp_user_item_dim=32,
        mlp_time_dim=8,
        # mlp_metadata_feature_dims=[],
        mlp_metadata_embedding_dims=[],
        num_mlp_layers=4,
        layers_ratio=2,
        dropout=0
    ):
        super(NCF, self).__init__()
        
        # Embedding layers
        self.mf_user_embedding = nn.Embedding(num_users, factors)
        self.mf_item_embedding = nn.Embedding(num_items, factors)
        
        self.mlp_user_embedding = nn.Embedding(num_users, mlp_user_item_dim)
        self.mlp_item_embedding = nn.Embedding(num_items, mlp_user_item_dim)
        
        self.time_embedding = nn.Linear(1, mlp_time_dim)
        
        # Metadata projection layers
        self.metadata_projection_layers = nn.ModuleList()
        for embed_dim in mlp_metadata_embedding_dims:
            self.metadata_projection_layers.append(nn.Linear(1, embed_dim))
        # for feature_dim, embed_dim in zip(mlp_metadata_feature_dims, mlp_metadata_embedding_dims):
        #     self.metadata_projection_layers.append(nn.Linear(feature_dim, embed_dim))
        total_metadata_embedding_dim = sum(mlp_metadata_embedding_dims)

        # MLP layers
        self.dropout = dropout
        self.mlp_layers = nn.ModuleList()
        
        mlp_input_size = mlp_user_item_dim * 2 + mlp_time_dim + total_metadata_embedding_dim
        for i in range(0, num_mlp_layers):
            mlp_output_size = int(mlp_input_size/layers_ratio)
            self.mlp_layers.append(nn.Linear(mlp_input_size, mlp_output_size))
            mlp_input_size = mlp_output_size
        
        self.prediction = nn.Linear(factors + mlp_output_size, 1)

    def forward(self, user, item, timestamp, metadata):
        mf_user_embed = self.mf_user_embedding(user)
        mf_item_embed = self.mf_item_embedding(item)
        mf_vector = torch.mul(mf_user_embed, mf_item_embed)
        
        mlp_user_embed = self.mlp_user_embedding(user)
        mlp_item_embed = self.mlp_item_embedding(item)
        mlp_vector = torch.cat([mlp_user_embed, mlp_item_embed], dim=-1)
        
        timestamp = timestamp.float().unsqueeze(1)
        time_embed = self.time_embedding(timestamp)
        mlp_vector = torch.cat([mlp_vector, time_embed], dim=-1)
        
        metadata_embeds = []
        for feature_values, projection_layer in zip(metadata, self.metadata_projection_layers):
            feature_values = feature_values.float().unsqueeze(1)
            metadata_embeds.append(projection_layer(feature_values))
        metadata_vector = torch.cat(metadata_embeds, dim=-1)
        mlp_vector = torch.cat([mlp_vector, metadata_vector], dim=-1)
        
        for layer in self.mlp_layers:
            mlp_vector = F.relu(layer(mlp_vector))
            if self.dropout > 0:
                mlp_vector = F.dropout(mlp_vector, p=self.dropout, training=self.training)
        
        vector = torch.cat([mf_vector, mlp_vector], dim=-1)
        logits = self.prediction(vector)
        output = torch.sigmoid(logits)
        
        return output.view(-1)
