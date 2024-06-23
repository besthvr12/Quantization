# my_model.py

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import logging
import collections
import json
from pytorch_quantization import calib
from pytorch_quantization import nn as quant_nn
from pytorch_quantization.tensor_quant import QuantDescriptor
from pytorch_quantization import quant_modules
def get_quantized_model(model_path):
    """
    Load and return the quantized model from the specified path.
    """
    model = torch.load(model_path)
    return model

def get_data_loader(data_path):
    """
    Load and return the data loader for the specified dataset.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    dataset = datasets.ImageFolder(root=data_path, transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    return data_loader

def get_optimizer(model):
    """
    Return the optimizer for the specified model.
    """
    Learning_Rate =0.001
    optimizer = optim.Adam(model.parameters(),lr=Learning_Rate)
    return optimizer

def get_criterion():
    """
    Return the loss criterion.
    """
    criterion = nn.CrossEntropyLoss()
    return criterion
