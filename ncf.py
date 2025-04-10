import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(
            self,
            num_users,
            num_items,
            embedding_dim=32,
            feature_dims=None,
            image_dim=None,
            dropout=0.0,
            layers_ratio=2,
            num_layers=4,
            gaussian_mean=0,
            gaussian_std=0.01
    ):
        super(MLP, self).__init__()

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        mlp_input_dim = embedding_dim * 2

        self.projection_layers = nn.ModuleDict()
        if feature_dims is not None:
            for feature, (input_dim, output_dim) in feature_dims.items():
                if input_dim == 1:
                    self.projection_layers[feature] = nn.Linear(input_dim, output_dim)
                else:
                    self.projection_layers[feature] = nn.EmbeddingBag(input_dim, output_dim, mode='mean')
                mlp_input_dim += output_dim

        if image_dim is not None:
            self.image_layer = nn.Linear(512, image_dim)
            mlp_input_dim += image_dim
        else:
            self.image_layer = None

        self.dropout = dropout
        self.layers = nn.ModuleList()

        for i in range(0, num_layers):
            mlp_output_dim = int(mlp_input_dim/layers_ratio)
            self.layers.append(nn.Linear(mlp_input_dim, mlp_output_dim))
            mlp_input_dim = mlp_output_dim

        self._init_weights(gaussian_mean, gaussian_std)

    def _init_weights(self, mean=0, std=0.01):
        """
        Initialize all weights using Normal distribution
        Args:
            mean: Mean of the Gaussian distribution
            std: Standard deviation of the Gaussian distribution
        """
        nn.init.normal_(self.user_embedding.weight, mean=mean, std=std)
        nn.init.normal_(self.item_embedding.weight, mean=mean, std=std)

        # if self.projection_layers is not None:
        #     for layer in self.projection_layers:
        #         if isinstance(layer, nn.Embedding):
        #             nn.init.normal_(layer.weight, mean=mean, std=std)

    def forward(self, user, item, features, image):
        user_embed = self.user_embedding(user)
        item_embed = self.item_embedding(item)
        vector = torch.cat([user_embed, item_embed], dim=-1)

        for feature, layer in self.projection_layers.items():
            if isinstance(layer, nn.Linear):
                feature_embed = layer(features[feature].unsqueeze(-1))
            else:
                indices, offsets = features[feature]
                feature_embed = layer(indices, offsets)
            vector = torch.cat([vector, feature_embed], dim=-1)

        if self.image_layer is not None:
            image_embed = self.image_layer(image)
            vector = torch.cat([vector, image_embed], dim=-1)

        for layer in self.layers:
            vector = F.relu(layer(vector))
            if self.dropout > 0:
                vector = F.dropout(vector, p=self.dropout, training=self.training)

        return vector

class GMF(nn.Module):
    def __init__(
        self,
        num_users,
        num_items,
        factors=32,
        gaussian_mean=0,
        gaussian_std=0.01
    ):
        super(GMF, self).__init__()

        self.user_embedding = nn.Embedding(num_users, factors)
        self.item_embedding = nn.Embedding(num_items, factors)

        self._init_weights(gaussian_mean, gaussian_std)

    def _init_weights(self, mean=0, std=0.01):
        """
        Initialize all weights using Normal distribution
        Args:
            mean: Mean of the Gaussian distribution
            std: Standard deviation of the Gaussian distribution
        """
        nn.init.normal_(self.user_embedding.weight, mean=mean, std=std)
        nn.init.normal_(self.item_embedding.weight, mean=mean, std=std)

    def forward(self, user, item):
        user_embed = self.user_embedding(user)
        item_embed = self.item_embedding(item)
        vector = torch.mul(user_embed, item_embed)
        return vector

class NCF(nn.Module):
    def __init__(
        self,
        num_users,
        num_items,
        factors=8,
        mlp_user_item_dim=32,
        mlp_feature_dims=None, # Dictionary with keys are features, values are tuples of (input, output),
        image_dim=None,
        num_mlp_layers=4,
        layers_ratio=2,
        dropout=0,
        gaussian_mean=0,
        gaussian_std=0.01
    ):
        super(NCF, self).__init__()

        self.gmf_module = GMF(num_users, num_items, factors, gaussian_mean, gaussian_std)
        
        self.mlp_module = MLP(
            num_users=num_users,
            num_items=num_items,
            embedding_dim=mlp_user_item_dim,
            feature_dims=mlp_feature_dims,
            image_dim=image_dim,
            dropout=dropout,
            layers_ratio=layers_ratio,
            num_layers=num_mlp_layers,
            gaussian_mean=gaussian_mean,
            gaussian_std=gaussian_std,
        )
        mlp_output_dim = self.mlp_module.layers[num_mlp_layers-1].out_features

        self.prediction = nn.Linear(factors + mlp_output_dim, 1)

    def forward(self, user, item, features, images):
        gmf_vector = self.gmf_module(user, item)
        mlp_vector = self.mlp_module(user, item, features, images)

        vector = torch.cat([gmf_vector, mlp_vector], dim=-1)
        logit = self.prediction(vector)
        output = torch.sigmoid(logit)

        return output.view(-1)
