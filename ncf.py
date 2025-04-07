import torch
import torch.nn as nn
import torch.nn.functional as F
import time

class NCF(nn.Module):
    def __init__(
        self,
        num_users,
        num_items,
        factors=8,
        mlp_user_item_dim=32,
        mlp_additional_features=None, # Dictionary with keys are features, values are tuples of (input, output)
        num_mlp_layers=4,
        layers_ratio=2,
        dropout=0,
        gaussian_mean=0,
        gaussian_std=0.01
    ):
        super(NCF, self).__init__()
        
        # Embedding layers
        self.mf_user_embedding = nn.Embedding(num_users, factors)
        self.mf_item_embedding = nn.Embedding(num_items, factors)
        
        self.mlp_user_embedding = nn.Embedding(num_users, mlp_user_item_dim)
        self.mlp_item_embedding = nn.Embedding(num_items, mlp_user_item_dim)
        mlp_input_size = mlp_user_item_dim * 2

        self.projection_layers = nn.ModuleDict()
        if mlp_additional_features is not None:
            for feature, (input_dim, output_dim) in mlp_additional_features.items():
                if input_dim == 1:
                    self.projection_layers[feature] = nn.Linear(input_dim, output_dim)
                else:
                    self.projection_layers[feature] = nn.EmbeddingBag(input_dim, output_dim, mode='mean')
                mlp_input_size += output_dim

        # MLP layers
        self.dropout = dropout
        self.mlp_layers = nn.ModuleList()

        mlp_output_size = 0
        for i in range(0, num_mlp_layers):
            mlp_output_size = int(mlp_input_size/layers_ratio)
            self.mlp_layers.append(nn.Linear(mlp_input_size, mlp_output_size))
            mlp_input_size = mlp_output_size
        
        self.prediction = nn.Linear(factors + mlp_output_size, 1)

        # Initialize weights using Gaussian distribution
        self._init_weights(gaussian_mean, gaussian_std)

    def _init_weights(self, mean=0, std=0.01):
        """
        Initialize all weights using Gaussian distribution
        Args:
            mean: Mean of the Gaussian distribution
            std: Standard deviation of the Gaussian distribution
        """
        # Initialize embedding layers
        nn.init.normal_(self.mf_user_embedding.weight, mean=mean, std=std)
        nn.init.normal_(self.mf_item_embedding.weight, mean=mean, std=std)
        nn.init.normal_(self.mlp_user_embedding.weight, mean=mean, std=std)
        nn.init.normal_(self.mlp_item_embedding.weight, mean=mean, std=std)

        # if self.projection_layers is not None:
        #     for layer in self.projection_layers:
        #         if isinstance(layer, nn.Embedding):
        #             nn.init.normal_(layer.weight, mean=mean, std=std)

    def forward(self, user, item, additional_features):
        mf_user_embed = self.mf_user_embedding(user)
        mf_item_embed = self.mf_item_embedding(item)
        mf_vector = torch.mul(mf_user_embed, mf_item_embed)

        mlp_user_embed = self.mlp_user_embedding(user)
        mlp_item_embed = self.mlp_item_embedding(item)
        mlp_vector = torch.cat([mlp_user_embed, mlp_item_embed], dim=-1)

        for feature, layer in self.projection_layers.items():
            if isinstance(layer, nn.Linear):
                feature_embed = layer(additional_features[feature].unsqueeze(-1))
            else:
                indices, offsets = additional_features[feature]
                feature_embed = layer(indices, offsets)
            mlp_vector = torch.cat([mlp_vector, feature_embed], dim=-1)

        for layer in self.mlp_layers:
            mlp_vector = F.relu(layer(mlp_vector))
            if self.dropout > 0:
                mlp_vector = F.dropout(mlp_vector, p=self.dropout, training=self.training)

        vector = torch.cat([mf_vector, mlp_vector], dim=-1)
        logit = self.prediction(vector)
        output = torch.sigmoid(logit)

        return output.view(-1)
