import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Binary symmetric channel
def bsc(input_layer, p_s = 0.2):
    output = torch.where(input_layer.to(device) > p_s , input_layer, -input_layer) 
    return output