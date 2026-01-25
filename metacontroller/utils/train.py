import torch
import torch.nn as nn

def xavier_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        # Use xavier_uniform_ or xavier_normal_
        torch.nn.init.xavier_uniform_(m.weight.data)
        # Initialize bias to zero if it exists
        if m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        # BatchNorm layers often have different initialization needs
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0)