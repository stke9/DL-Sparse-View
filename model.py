import numpy as np
from networks import *
import torch
from torch.utils.data import TensorDataset, DataLoader
from util import *
import wandb
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--directory', type=str, default='default',
                        help='Stores the progress output in the directory name given')
args, unknown = parser.parse_known_args()

run_name = args.directory
wandb.init(project='DL-Sparse-View', name=run_name, entity='tldr-group')

print('Loading dataset...')
x, y = (torch.tensor(np.load('training_data/{}.npy'.format(pth)))
            for pth in ['FBP128', 'Phantom', ])

batch_size = 2
workers = 0
print('Preprocessing...')

prev_model_path = 'prev_model.pt'

# x_test, y_test = x[3975:, :, 128:256, 128:256], y[3975:, :, 128:256, 128:256]
# train_data = TensorDataset(x[:3975, :, 128:256, 128:256], y[:3975, :, 128:256, 128:256])
# train_data = DataLoader(train_data, batch_size=batch_size,
#                         shuffle=True, num_workers=workers)

x_test, y_test = x[3995:, :, :256, :256], y[3995:, :, :256, :256]
train_data = TensorDataset(x[:3975, :, :256, :256], y[:3975,:, :256, :256])
train_data = DataLoader(train_data, batch_size=batch_size,
                        shuffle=True, num_workers=workers)

# load the previous model
prev_model = DnCNN_OHE_res(in_ch=1, out_ch=5, depth=36, ch=128)
prev_model.load_state_dict(torch.load(prev_model_path))
# load current model
model = DnCNN_OHE_res(in_ch=7, out_ch=5, depth=12, ch=128)
opt = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.9))
prev_model.cuda()
prev_model.eval()
model.cuda()
loss_func = torch.nn.MSELoss()
print('Starting training loop...')

for epoch in range(200):
    for i, (x_train, y_train) in enumerate(train_data):
        with torch.no_grad():
            one_hot_round, img, prev_loss = prev_model(x_train.cuda(), y_train.cuda())
            noise = img - y_train.cuda()
            x_input = torch.cat((x_train.cuda(), img, one_hot_round, noise), 1)
        model.zero_grad()
        # x_train = prep_img(x_train)
        noise, img, loss = model(x_input, y_train.cuda())
        loss.backward()
        opt.step()
        if i % 30 == 0:
            with torch.no_grad():
                one_hot_round, img, prev_loss = prev_model(x_test.cuda(), y_test.cuda())
                noise = img - y_test.cuda()
                x_input = torch.cat((x_test.cuda(), img, one_hot_round, noise), 1)
                noise, img, t_loss = model(x_input, y_test.cuda(), post_proc=True)
                wandb.log({'test_loss': t_loss.item()**0.5,
                           'training_loss': loss.item()**0.5})
            print(epoch, i, loss.item()**0.5)
            images = [im[j, -1].cpu() for j in range(5) for im in [x_test, noise, img, y_test, abs(img-y_test.cuda())] ]
            wandb.log({"examples" : [wandb.Image(i) for i in images]})
    if (epoch%10) == 0:
        torch.save(model.state_dict(), run_name +'.pt')


