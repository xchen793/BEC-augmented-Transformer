import torch

checkpoint = "project2_model.pt"
print('checkpoint:', checkpoint)
ckpt = torch.load(checkpoint) 
# print('ckpt', ckpt)
transformer_sd = ckpt['net']
optimizer_sd = ckpt['opt'] 
lr_scheduler_sd = ckpt['lr_scheduler']

print(transformer_sd)
print(optimizer_sd)
print(lr_scheduler_sd)
