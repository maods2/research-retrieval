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
                

    def forward(self, *args: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """
        Forward pass through n-branch MLP.

        If single tensor provided: replicate to all branches
        If n tensors provided: pass each through corresponding branch

        Args:
            *args: Either one tensor (B, input_dim) or n_branches tensors of shape (B, input_dim)

        Returns:
            Tuple of tensors, one per branch, each of shape (B, d_k)
        """
        # Handle single input replication
        if len(args) == 1:
            x = args[0]
            if len(x.shape) != 2:
                raise ValueError(f"Invalid shape for input matrix: expected (N, d), got {x.shape}")
            out_tensors = []
            for mlp in self.mlps:
                out = x
                for layer in mlp:
                    out = layer(out)
                out_tensors.append(out)
            return tuple(out_tensors)
        
        # Handle multiple inputs (one per branch)
        elif len(args) == len(self.mlps):
            out_tensors = []
            for mlp, x in zip(self.mlps, args):
                if len(x.shape) != 2:
                    raise ValueError(f"Invalid shape for input matrix: expected (N, d), got {x.shape}")
                
                out = x
                for layer in mlp:
                    out = layer(out)
                out_tensors.append(out)
            return tuple(out_tensors)
        
        else:
            raise ValueError(f"Expected either 1 or {len(self.mlps)} tensors, got {len(args)}")
