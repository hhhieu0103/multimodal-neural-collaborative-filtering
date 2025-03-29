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
        mlp_time_dim=None,
        mlp_metadata_feature_dims=None,
        mlp_metadata_embedding_dims=None,
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
        mlp_input_size = mlp_user_item_dim * 2

        self.use_time = mlp_time_dim is not None
        if self.use_time:
            self.time_embedding = nn.Linear(1, mlp_time_dim)
            mlp_input_size += mlp_time_dim

        self.use_metadata = mlp_metadata_feature_dims is not None and mlp_metadata_embedding_dims is not None
        # Metadata projection layers
        if self.use_metadata:
            self.metadata_projection_layers = nn.ModuleList()
            for (feature_dim, embed_dim) in zip(mlp_metadata_feature_dims, mlp_metadata_embedding_dims):
                self.metadata_projection_layers.append(nn.Linear(feature_dim, embed_dim))
            mlp_input_size += sum(mlp_metadata_embedding_dims)

        # MLP layers
        self.dropout = dropout
        self.mlp_layers = nn.ModuleList()

        mlp_output_size = 0
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

        if self.use_time and timestamp is not None:
            timestamp = timestamp.float().unsqueeze(1) if timestamp.dim() == 1 else timestamp
            time_embed = self.time_embedding(timestamp)
            mlp_vector = torch.cat([mlp_vector, time_embed], dim=-1)

        if self.use_metadata and metadata is not None:
            metadata_embeds = []
            for i, (feature_values, projection_layer) in enumerate(zip(metadata, self.metadata_projection_layers)):
                if feature_values is not None:  # Check if feature values exist
                    # Ensure feature_values has the right shape
                    feature_values = feature_values.float()

                    # Debug shape issues
                    original_shape = feature_values.shape

                    # Reshape if needed - ensure 2D for linear layer
                    if len(feature_values.shape) > 2:
                        # If it's a 3D+ tensor (batch x items x features), reshape
                        batch_size = feature_values.shape[0]
                        feature_values = feature_values.reshape(batch_size, -1)
                    elif len(feature_values.shape) == 1:
                        # If it's a 1D tensor, add batch dimension
                        feature_values = feature_values.unsqueeze(0)

                    try:
                        # Apply projection
                        feature_embed = projection_layer(feature_values)
                        metadata_embeds.append(feature_embed)
                    except RuntimeError as e:
                        print(f"Error processing feature {i}: Original shape {original_shape}, "
                              f"Reshaped: {feature_values.shape}, "
                              f"Projection layer weight shape: {projection_layer.weight.shape}")
                        raise e

            if metadata_embeds:
                metadata_vector = torch.cat(metadata_embeds, dim=-1)
                mlp_vector = torch.cat([mlp_vector, metadata_vector], dim=-1)

        for layer in self.mlp_layers:
            mlp_vector = F.relu(layer(mlp_vector))
            if self.dropout > 0:
                mlp_vector = F.dropout(mlp_vector, p=self.dropout, training=self.training)

        vector = torch.cat([mf_vector, mlp_vector], dim=-1)
        logit = self.prediction(vector)
        output = torch.sigmoid(logit)

        return output.view(-1)
