import torch
import torch.nn as nn
import torch.nn.functional as F
import enum

class ModelType(enum.Enum):
    EARLY_FUSION = 1
    LATE_FUSION = 2

class MLPFull(nn.Module):
    def __init__(
            self,
            num_users,
            num_items,
            embedding_dim=32,
            feature_dims=None,
            image_dim=None,
            audio_dim=None,
            dropout=0.0,
            layers_ratio=2,
            num_layers=4,
            gaussian_mean=0,
            gaussian_std=0.01
    ):
        super(MLPFull, self).__init__()

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        nn.init.normal_(self.user_embedding.weight, mean=gaussian_mean, std=gaussian_std)
        nn.init.normal_(self.item_embedding.weight, mean=gaussian_mean, std=gaussian_std)
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

        if audio_dim is not None:
            self.audio_layer = nn.Linear(128, audio_dim)
            mlp_input_dim += audio_dim
        else:
            self.audio_layer = None

        self.dropout = dropout
        self.layers = nn.ModuleList()

        for i in range(0, num_layers):
            mlp_output_dim = int(mlp_input_dim/layers_ratio)
            self.layers.append(nn.Linear(mlp_input_dim, mlp_output_dim))
            mlp_input_dim = mlp_output_dim

    def forward(self, user, item, features, image, audio):
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

        if self.audio_layer is not None:
            audio_embed = self.audio_layer(audio)
            vector = torch.cat([vector, audio_embed], dim=-1)

        for layer in self.layers:
            vector = F.relu(layer(vector))
            if self.dropout > 0:
                vector = F.dropout(vector, p=self.dropout, training=self.training)

        return vector


class MLP(nn.Module):
    def __init__(
            self,
            num_users,
            num_items,
            embedding_dim=32,
            dropout=0.0,
            layers_ratio=2,
            num_layers=4,
            gaussian_mean=0,
            gaussian_std=0.01
    ):
        super(MLP, self).__init__()

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        nn.init.normal_(self.user_embedding.weight, mean=gaussian_mean, std=gaussian_std)
        nn.init.normal_(self.item_embedding.weight, mean=gaussian_mean, std=gaussian_std)

        self.dropout = dropout
        self.layers = nn.ModuleList()

        mlp_input_dim = embedding_dim * 2
        for i in range(0, num_layers):
            mlp_output_dim = int(mlp_input_dim / layers_ratio)
            self.layers.append(nn.Linear(mlp_input_dim, mlp_output_dim))
            mlp_input_dim = mlp_output_dim

    def forward(self, user, item):
        user_embed = self.user_embedding(user)
        item_embed = self.item_embedding(item)
        vector = torch.cat([user_embed, item_embed], dim=-1)

        for layer in self.layers:
            vector = F.relu(layer(vector))
            if self.dropout > 0:
                vector = F.dropout(vector, p=self.dropout, training=self.training)

        return vector


class MLPMetadata(nn.Module):
    def __init__(
            self,
            num_users,
            num_items,
            feature_dims,
            embedding_dim=32,
            dropout=0.0,
            layers_ratio=2,
            num_layers=4,
            gaussian_mean=0,
            gaussian_std=0.01
    ):
        super(MLPMetadata, self).__init__()

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        nn.init.normal_(self.user_embedding.weight, mean=gaussian_mean, std=gaussian_std)
        nn.init.normal_(self.item_embedding.weight, mean=gaussian_mean, std=gaussian_std)
        mlp_input_dim = embedding_dim * 2

        self.projection_layers = nn.ModuleDict()
        for feature, (input_dim, output_dim) in feature_dims.items():
            if input_dim == 1:
                self.projection_layers[feature] = nn.Linear(input_dim, output_dim)
            else:
                self.projection_layers[feature] = nn.EmbeddingBag(input_dim, output_dim, mode='mean')
            mlp_input_dim += output_dim

        self.dropout = dropout
        self.layers = nn.ModuleList()

        for i in range(0, num_layers):
            mlp_output_dim = int(mlp_input_dim / layers_ratio)
            self.layers.append(nn.Linear(mlp_input_dim, mlp_output_dim))
            mlp_input_dim = mlp_output_dim

    def forward(self, user, item, features):
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

        for layer in self.layers:
            vector = F.relu(layer(vector))
            if self.dropout > 0:
                vector = F.dropout(vector, p=self.dropout, training=self.training)

        return vector


class MLPImage(nn.Module):
    def __init__(
            self,
            num_items,
            image_dim,
            embedding_dim=32,
            dropout=0.0,
            layers_ratio=2,
            num_layers=4,
            gaussian_mean=0,
            gaussian_std=0.01
    ):
        super(MLPImage, self).__init__()

        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        nn.init.normal_(self.item_embedding.weight, mean=gaussian_mean, std=gaussian_std)
        mlp_input_dim = embedding_dim

        self.image_bn = nn.BatchNorm1d(512)
        self.image_layer = nn.Linear(512, image_dim)
        mlp_input_dim += image_dim

        self.dropout = dropout
        self.layers = nn.ModuleList()

        for i in range(0, num_layers):
            mlp_output_dim = int(mlp_input_dim / layers_ratio)
            self.layers.append(nn.Linear(mlp_input_dim, mlp_output_dim))
            mlp_input_dim = mlp_output_dim

    def forward(self, item, image):
        item_embed = self.item_embedding(item)

        image_feature = self.image_bn(image)
        image_embed = self.image_layer(image_feature)
        vector = torch.cat([item_embed, image_embed], dim=-1)

        for layer in self.layers:
            vector = F.relu(layer(vector))
            if self.dropout > 0:
                vector = F.dropout(vector, p=self.dropout, training=self.training)

        return vector


class MLPAudio(nn.Module):
    def __init__(
            self,
            num_users,
            num_items,
            audio_dim,
            embedding_dim=32,
            dropout=0.0,
            layers_ratio=2,
            num_layers=4,
            gaussian_mean=0,
            gaussian_std=0.01
    ):
        super(MLPAudio, self).__init__()

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        nn.init.normal_(self.user_embedding.weight, mean=gaussian_mean, std=gaussian_std)
        nn.init.normal_(self.item_embedding.weight, mean=gaussian_mean, std=gaussian_std)
        mlp_input_dim = embedding_dim * 2

        self.audio_layer = nn.Linear(128, audio_dim)
        mlp_input_dim += audio_dim

        self.dropout = dropout
        self.layers = nn.ModuleList()

        for i in range(0, num_layers):
            mlp_output_dim = int(mlp_input_dim / layers_ratio)
            self.layers.append(nn.Linear(mlp_input_dim, mlp_output_dim))
            mlp_input_dim = mlp_output_dim

    def forward(self, user, item, audio):
        user_embed = self.user_embedding(user)
        item_embed = self.item_embedding(item)
        vector = torch.cat([user_embed, item_embed], dim=-1)

        audio_embed = self.audio_layer(audio)
        vector = torch.cat([vector, audio_embed], dim=-1)

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
        audio_dim=None,
        num_mlp_layers=4,
        layers_ratio=2,
        dropout=0,
        gaussian_mean=0,
        gaussian_std=0.01,
        model_type=ModelType.EARLY_FUSION
    ):
        super(NCF, self).__init__()
        self.model_type = model_type

        self.gmf_module = GMF(num_users, num_items, factors, gaussian_mean, gaussian_std)

        self.mlp_module = MLPFull(
            num_users=num_users,
            num_items=num_items,
            embedding_dim=mlp_user_item_dim,
            feature_dims=mlp_feature_dims,
            image_dim=image_dim,
            audio_dim=audio_dim,
            dropout=dropout,
            layers_ratio=layers_ratio,
            num_layers=num_mlp_layers,
            gaussian_mean=gaussian_mean,
            gaussian_std=gaussian_std,
        )
        mlp_output_dim = self.mlp_module.layers[-1].out_features

        if model_type == ModelType.LATE_FUSION:
            if mlp_feature_dims is None and image_dim is None and audio_dim is None:
                raise ValueError('MLP feature dims or image dim or audio dim must be specified for late fusion model')

            mlp = MLP(
                num_users=num_users,
                num_items=num_items,
                embedding_dim=mlp_user_item_dim,
                dropout=dropout,
                layers_ratio=layers_ratio,
                num_layers=num_mlp_layers,
                gaussian_mean=gaussian_mean,
                gaussian_std=gaussian_std,
            )
            self.mlp_module = nn.ModuleDict({'mlp': mlp})
            mlp_output_dim = mlp.layers[-1].out_features

            self.mlp_module['mlp_metadata'] = None
            if mlp_feature_dims is not None:
                mlp_metadata = MLPMetadata(
                    num_users=num_users,
                    num_items=num_items,
                    embedding_dim=mlp_user_item_dim,
                    feature_dims=mlp_feature_dims,
                    dropout=dropout,
                    layers_ratio=layers_ratio,
                    num_layers=num_mlp_layers,
                    gaussian_mean=gaussian_mean,
                    gaussian_std=gaussian_std,
                )
                self.mlp_module['mlp_metadata'] = mlp_metadata
                mlp_output_dim += mlp_metadata.layers[-1].out_features

            self.mlp_module['mlp_image'] = None
            if image_dim is not None:
                mlp_image = MLPImage(
                    num_items=num_items,
                    embedding_dim=mlp_user_item_dim,
                    image_dim=image_dim,
                    dropout=dropout,
                    layers_ratio=layers_ratio,
                    num_layers=num_mlp_layers
                )
                self.mlp_module['mlp_image'] = mlp_image
                mlp_output_dim += mlp_image.layers[-1].out_features

            self.mlp_module['mlp_audio'] = None
            if audio_dim is not None:
                mlp_audio = MLPAudio(
                    num_users=num_users,
                    num_items=num_items,
                    audio_dim=audio_dim,
                    embedding_dim=mlp_user_item_dim,
                    dropout=dropout,
                    layers_ratio=layers_ratio,
                    num_layers=num_mlp_layers,
                )
                self.mlp_module['mlp_audio'] = mlp_audio
                mlp_output_dim += mlp_audio.layers[-1].out_features

        self.prediction = nn.Linear(factors + mlp_output_dim, 1)

    def forward(self, user, item, features, images, audio):
        gmf_vector = self.gmf_module(user, item)
        if self.model_type == ModelType.EARLY_FUSION:
            mlp_vector = self.mlp_module(user, item, features, images, audio)
        else:
            mlp_vector = self.mlp_module['mlp'](user, item)

            mlp_metadata = self.mlp_module['mlp_metadata']
            if mlp_metadata is None and features is not None:
                raise ValueError('MLP feature dims is missing')
            elif mlp_metadata is not None and features is None:
                raise ValueError('MLP feature data is missing')
            elif mlp_metadata is not None and features is not None:
                mlp_metadata_vector = self.mlp_module['mlp_metadata'](user, item, features)
                mlp_vector = torch.cat([mlp_vector, mlp_metadata_vector], dim=-1)

            mlp_image = self.mlp_module['mlp_image']
            if mlp_image is None and images is not None:
                raise ValueError('MLP image dim is missing')
            elif mlp_image is not None and images is None:
                raise ValueError('MLP image data is missing')
            elif mlp_image is not None and images is not None:
                mlp_image_vector = self.mlp_module['mlp_image'](item, images)
                mlp_vector = torch.cat([mlp_vector, mlp_image_vector], dim=-1)

            mlp_audio = self.mlp_module['mlp_audio']
            if mlp_audio is None and audio is not None:
                raise ValueError('MLP audio dim is missing')
            elif mlp_audio is not None and audio is None:
                raise ValueError('MLP audio data is missing')
            elif mlp_audio is not None and audio is not None:
                mlp_audio_vector = self.mlp_module['mlp_audio'](user, item, audio)
                mlp_vector = torch.cat([mlp_vector, mlp_audio_vector], dim=-1)

        vector = torch.cat([gmf_vector, mlp_vector], dim=-1)
        logit = self.prediction(vector)
        output = torch.sigmoid(logit)

        return output.view(-1)
