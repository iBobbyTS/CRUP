import torch

x = torch.load('/Users/ibobby/Downloads/ckpt_2.pth', map_location=torch.device('cpu'))
k = list(x.keys())
k.remove('weight')
y = {
    'weight': x['weight'],
    'model_params': {}
}
for i in k:
    y['model_params'][i] = x[i]
torch.save(y, '/Users/ibobby/Downloads/ckpt_2_.pth')
