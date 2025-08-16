from src.core.base_model import BaseModel

import torch
import torch.nn as nn


# Local registry to avoid circular import
try:
    from src.factories.model_factory import register_model
except ImportError:
    register_model = lambda name: (lambda cls: cls)


@register_model('mynet')
class MyNet(nn.Module, BaseModel):
    def __init__(self, input_dim=2048, output_dim=10, **kwargs):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
