import torch
import torch.nn as nn

class NCF(nn.Module):
    def __init__(
        self,
        num_users,
        num_items,
        factors=8,
        layers=[64, 32, 16, 8],
    ):
        super(NCF, self).__init__()
        
        # Embedding layers
        self.mf_user_embedding = nn.Sequential(
            nn.Embedding(num_users, factors),
            nn.Flatten()
        )
        
        self.mf_item_embedding = nn.Sequential(
             nn.Embedding(num_items, factors),
             nn.Flatten()
        )
        
        self.mlp_user_embedding = nn.Sequential(
            nn.Embedding(num_users, int(layers[0]/2)),
            nn.Flatten()
        )
         
        self.mlp_item_embedding = nn.Sequential(
            nn.Embedding(num_items, int(layers[0]/2)),
            nn.Flatten()
        )

        # MLP layers
        mlp_layers = []
        for idx, dim in enumerate(layers[:-1]):
            mlp_layers.append(nn.Linear(int(dim), int(layers[idx + 1])))
            mlp_layers.append(nn.ReLU())
        self.mlp_layers = nn.Sequential(*mlp_layers)
        
        self.logit = nn.Linear(factors + layers[-1], 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, user, item):
        mf_user_embed = self.mf_user_embedding(user)
        mf_item_embed = self.mf_item_embedding(item)
        
        mlp_user_embed = self.mlp_user_embedding(user)
        mlp_item_embed = self.mlp_item_embedding(item)

        mf_vector = torch.mul(mf_user_embed, mf_item_embed)
        mlp_vector = torch.concat([mlp_user_embed, mlp_item_embed], -1)
        mlp_vector = self.mlp_layers(mlp_vector)
        
        final_vector = torch.concat([mf_vector, mlp_vector], dim=-1)
        logit = self.logit(final_vector)
        output = self.sigmoid(logit)
        return output.view(-1)
