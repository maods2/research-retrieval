from typing import Any
import itertools

import torch
import torch.nn as nn

# Local registry to avoid circular import
try:
    from factories.model_factory import register_model
except ImportError:
    register_model = lambda name: (lambda cls: cls)

@register_model('n_branch_mlp')
class N_BranchMLP(nn.Module):
    """
    MLP network with n separate, independent branches
        with their own weights and weight updates

    No activation on the last layer
    """
    def __init__(self, model_config: dict[str, Any], dtype: torch.dtype = torch.float32):
        super(N_BranchMLP, self).__init__()
        if model_config.get('dropout') is not None and len(model_config['dropout']) != len(model_config['mlp_layers']):
            raise ValueError(f"Dropout vector does not match layer sequence: got {model_config['dropout']} and {model_config['mlp_layers']}")

        self.n_branches = model_config.get('n_mlp_branches') or model_config['model_code'].split('_')[0]
        self.input_dim  = model_config['encoder_dim']
        self.mlp_layers = model_config['mlp_layers']
        self.activation = model_config['mlp_activation']
        self.d_k        = model_config['mlp_layers'][-1]
        self.dropout    = model_config.get('dropout')
        self.activation_params =  model_config.get('mlp_activation_params') or {}
        self.mlp_activation = model_config.get('mlp_activation')
        self.dtype      = dtype

        self.mlps = nn.ModuleList()
        for _ in range(model_config['n_mlp_branches']):
            mlp = nn.ModuleList()
            for layer, (d1, d2) in enumerate(itertools.pairwise([self.input_dim, *self.mlp_layers])):
                # add normalization before every linear layer
                # to guarantee features won't become too small (mean->0) or too large (in absolute value)
                mlp.append(nn.BatchNorm1d(num_features=d1))

                mlp.append(nn.Linear(d1, d2, dtype=dtype))
                if layer == len(self.mlp_layers) - 1:
                    mlp.append(nn.BatchNorm1d(num_features=d2))
                elif self.activation == "relu":
                    mlp.append(nn.ReLU(**self.activation_params))
                elif self.activation == "sigmoid":
                    mlp.append(nn.Sigmoid(**self.activation_params))
                elif self.activation == "leaky_relu":
                    mlp.append(nn.LeakyReLU(**self.activation_params))
                else:
                    raise ValueError(f"Invalid activation function for MLP: {self.mlp_activation}")
                if self.dropout is not None:
                    mlp.append(nn.Dropout(p=self.dropout[layer]))
            self.mlps.append(mlp)
                

    def forward(self, *args: torch.Tensor) -> list[torch.Tensor]:
        """
        Calculate deep similarity function.

        :param torch.Tensor x: matrix (N, d) of embeddings
        :return: matrix (N, d_k) of transformed embeddings
        """
        if len(args) != len(self.mlps):
                raise ValueError(f"Insufficient tensors to forward function: expected {len(self.mlps)} tensors (N, d)")
            
        out_tensors = []
        for mlp, x in zip(self.mlps, args):
            if len(x.shape) != 2:
                raise ValueError("Invalid shape for input matrix:"
                                f" expected (N, d), got {x.shape}")
            
            # apply MLP
            for layer in mlp:
                x = layer(x)
            out_tensors.append(x)

        return out_tensors
