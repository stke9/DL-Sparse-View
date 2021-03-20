import numpy as np
from networks import *
import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

print('Loading dataset...')
x, y = (torch.tensor(np.load('training_data/{}.npy'.format(pth)))
            for pth in ['FFBP128', 'Phantom', ])

batch_size = 25
workers = 0
print('Preprocessing...')
train_data = TensorDataset(x[:3750, :, 128:256, 128:256 ], y[:3750, :, 128:256, 128:256])
train_data = DataLoader(train_data, batch_size=batch_size,
                        shuffle=True, num_workers=workers)
test_data = TensorDataset(x[:3750], y[:3750])
test_data = DataLoader(test_data, batch_size=batch_size,
                        shuffle=True, num_workers=workers)

model = DnCNN()
opt = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
model.cuda()
loss_func = torch.nn.MSELoss()
print('Starting training loop...')
losses = []
for epoch in range(30):
    for i, (x_train, y_train) in enumerate(train_data):
        model.zero_grad()
        x_train = x_train.cuda()
        noise = model(x_train)
        img = x_train - noise
        loss = loss_func(img, y_train.cuda())
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if i % 50 == 0:
            print(epoch, i, loss.item())
            plt.plot(losses[50:])
            plt.savefig('losses.png')
            plt.close()
            fig, axs = plt.subplots(2, 2)
            for im, ax in zip([x_train, noise, img, y_train], axs.ravel()):
                ax.imshow(im[0, 0].cpu().detach())
            plt.savefig('test.png')
            plt.close()


