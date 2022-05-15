import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Binary Erasure Channel
def bec(input_layer, p_e = 0.2):
    m = nn.Dropout(p=p_e)
    output = m(input_layer)*(1-p_e)
    return output

def combined_bec_bsc(input_layer, p_e = 0.2, p_s = 0.5):
    m = nn.Dropout(p=p_e)
    output = m(input_layer)*(1-p_e)
    output = torch.where(input_layer.to(device) > p_s , input_layer, -input_layer) 
    return output


def combined_bsc_bec(input_layer, p_e = 0.2, p_s = 0.5):
    output = torch.where(input_layer.to(device) > p_s , input_layer, -input_layer) 
    m = nn.Dropout(p=p_e)
    output = m(input_layer)*(1-p_e)
    return output
