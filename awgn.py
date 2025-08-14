import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Additive White Gaussian Noise Channel
def gaussian_noise_layer(input_layer):
    output = input_layer + torch.randn(size = input_layer.size()).to(device)
    return output