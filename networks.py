import torch.nn as nn
from util import *
import wandb

class DnCNN(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, depth=18, ch=64):
        super(DnCNN, self).__init__()
        self.depth = depth
        # in layer
        self.loss_func = nn.MSELoss()
        self.conv1 = nn.Conv2d(in_channels=in_ch, out_channels=ch, kernel_size=3, padding=1, padding_mode='replicate', bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        # hidden layers
        hidden_layers = []
        for i in range(depth):
            hidden_layers.append(nn.Conv2d(in_channels=ch, out_channels=ch, kernel_size=3, padding=1, padding_mode='replicate', bias=False))
            hidden_layers.append(nn.GroupNorm(4, ch))
            hidden_layers.append(nn.ReLU(inplace=True))
        self.mid_layer = nn.Sequential(*hidden_layers)
        # out layer
        self.conv3 = nn.Conv2d(in_channels=ch, out_channels=out_ch, kernel_size=3, padding=1, padding_mode='replicate', bias=False)


    def forward_base(self, x):
        out = self.relu1(self.conv1(x))
        out = self.mid_layer(out)
        out = self.conv3(out)
        return out

    def forward(self, x, y):
        noise = self.forward_base(x)
        img = x - noise
        loss = self.loss_func(img, y)
        return noise, img, loss

    def predict(self, x):
        with torch.no_grad():
            noise = self.forward_base(x)
            img = x - noise
            return img

class DnCNN_OHE(DnCNN):
    def __init__(self, in_ch=1, out_ch=1, depth=18, ch=64):
        super(DnCNN_OHE, self).__init__(in_ch, out_ch, depth, ch)

    def forward(self, x, y, post_proc=False):
        y_prep = one_hot_y(y).cuda()
        noise = self.forward_base(x)
        one_hot_pred = torch.softmax(noise[:, 1:], dim=1)
        loss_one_hot = self.loss_func(one_hot_pred, y_prep)
        img = (x[:, 0] - noise[:, 0]).unsqueeze(1)
        loss_noise = (((img - y)**2) * one_hot_pred[:, -1]).mean()
        loss = 1000 * loss_noise + loss_one_hot
        one_hot_round = torch.round(one_hot_pred)
        wandb.log({'oh': loss_one_hot,
                   'noise': loss_noise})
        if post_proc:
            img = img[:, 0]
            img[one_hot_round[:, 0]==1] = 0
            img[one_hot_round[:, 1]==1] = 0.194
            img[one_hot_round[:, 2]==1] = 0.233
            loss_pred = self.loss_func(img, y[:, 0])
            wandb.log({'loss_pred': loss_pred.item()**0.5})
            img = img.unsqueeze(1)
        return one_hot_round, img, loss

    def predict(self, x, y=None):
        with torch.no_grad():
            noise = self.forward_base(x)
            one_hot_pred = torch.softmax(noise[:, 1:], dim=1)
            img = x[:, 0] - noise[:, 0].unsqueeze(1)
            one_hot_round = torch.round(one_hot_pred)
            img = img[:, 0]
            img[one_hot_round[:, 0]==1] = 0
            img[one_hot_round[:, 1]==1] = 0.194
            img[one_hot_round[:, 2]==1] = 0.233
            if y:
                loss_pred = self.loss_func(img, y[:, 0])
                return img, loss_pred
            return img

class DnCNN_OHE_res(DnCNN_OHE):
    def __init__(self, in_ch=1, out_ch=1, depth=18, ch=64):
        super(DnCNN_OHE_res, self).__init__(in_ch, out_ch, depth, ch)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(depth):
            self.convs.append(nn.Conv2d(in_channels=ch, out_channels=ch, kernel_size=3, padding=1, padding_mode='replicate', bias=False))
            self.norms.append(nn.BatchNorm2d(ch))

    def forward_base(self, x):
        out = self.relu1(self.conv1(x))
        for i in range(self.depth//2):
            j = 2*i
            conv1, norm1 = self.convs[j], self.norms[j]
            conv2, norm2 = self.convs[j+1], self.norms[j+1]
            if i < self.depth//2 - 1:
                side = norm1(conv1(self.relu1(norm2(conv2(out)))))
                out = self.relu1(out + side)
            else:
                out = self.relu1(norm1(conv1(out)))
        out = self.conv3(out)
        return out