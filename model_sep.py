import numpy as np
from networks import *
import torch
from torch.utils.data import TensorDataset, DataLoader
from util import *
import wandb
import os

os.environ['WANDB_SILENT']="true"

print('Loading dataset...')
x, y = (torch.tensor(np.load('training_data/{}.npy'.format(pth)))
        for pth in ['FBP128', 'Phantom', ])

batch_size = 2
workers = 0
print('Preprocessing...')

# x_test, y_test = x[3975:, :, 128:256, 128:256], y[3975:, :, 128:256, 128:256]
# train_data = TensorDataset(x[:3975, :, 128:256, 128:256], y[:3975, :, 128:256, 128:256])
# train_data = DataLoader(train_data, batch_size=batch_size,
#                         shuffle=True, num_workers=workers)

x_test, y_test = x[3985:, :, :256, :256], y[3985:, :, :256, :256]
x_rot, y_rot = x[:3975, :, :256, :256], y[:3975, :, :256, :256]
for q in range(1,4):
    x_rot = torch.cat((x_rot, torch.rot90(x, q, dims=(2, 3))[:3975, :, :256, :256]))
    y_rot = torch.cat((y_rot, torch.rot90(y, q, dims=(2, 3))[:3975, :, :256, :256]))
print('Dataloading')
train_data = TensorDataset(x_rot, y_rot)
train_data = DataLoader(train_data, batch_size=batch_size,
                        shuffle=True, num_workers=workers)
depth = 18
ch = 128
run_name = '{}lay_{}ch_7k_ker3'.format(depth, ch)
print('initing wandb')
wandb.init(project='DL-Sparse-View', name=run_name, entity='tldr-group')

model = DnCNN_Pure_OHE(in_ch=1, out_ch=4, k=5, p=2, depth=depth, ch=ch)
model = nn.DataParallel(model, list(range(1)))
opt = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.9))
model.cuda()
loss_func = torch.nn.MSELoss()
print('Starting training loop...')

for epoch in range(15):
    for i, (x_train, y_train) in enumerate(train_data):
        model.zero_grad()
        loss, _, _ = model(x_train.cuda(), y_train.cuda())
        loss = loss.mean()
        loss.backward()
        opt.step()
        if i % 30 == 0:
            with torch.no_grad():
                t_loss, y_prep, one_hot_pred = model(x_test.cuda(), y_test.cuda(), post_proc=True)
            # m denotes max channel
            m = torch.argmax(one_hot_pred, 1)
            for ch in range(4):
                # make one_hot_pred a true ohe
                one_hot_pred[:, ch][m == ch] = 1
                one_hot_pred[:, ch][m != ch] = 0
            for ch in range(4):
                fp = len(one_hot_pred[:, ch][(one_hot_pred[:, ch] == 1) & (y_prep[:, ch] == 0)])
                fn = len(one_hot_pred[:, ch][(one_hot_pred[:, ch] == 0) & (y_prep[:, ch] != 0)])
                wandb.log({'false_positive_channel{}'.format(ch): fp})
                wandb.log({'false_negative_channel{}'.format(ch): fn})
            # assign x values as we would in next model
            x_test[:, 0][one_hot_pred[:, 0] == 1] = 0
            x_test[:, 0][one_hot_pred[:, 1] == 1] = 0.194
            x_test[:, 0][one_hot_pred[:, 2] == 1] = 0.233
            x_test[:, 0][one_hot_pred[:, 3] == 1] = y_test[:, 0][one_hot_pred[:, 3] == 1]
            loss = loss_func(x_test, y_test)
            wandb.log({'effective loss': loss.item()**0.5})
            print(epoch, i, loss.mean().item()**0.5)
            images = [im[j].cpu() for j in range(4) for im in [x_test, y_test]]
            wandb.log({"examples" : [wandb.Image(i) for i in images]})
            torch.save(model.state_dict(), run_name +'.pt')


