import torch
from networks import *
import numpy as np
from scipy.ndimage import rotate as rot
import matplotlib.pyplot as plt

# prev_model
prev_model = DnCNN_OHE_res(in_ch=1, out_ch=5, depth=36, ch=128)
prev_model.load_state_dict(torch.load('prev_model.pt'))
# cur model
model = DnCNN_OHE_res(in_ch=2, out_ch=5, depth=48, ch=128)
model.load_state_dict(torch.load('48deep_with_input_from_36deep.pt'))
prev_model.eval()
prev_model.cuda()
model.eval()
model.cuda()

x_test = torch.tensor(np.load('validation_data/FBP128_validation.npy')).unsqueeze(1)
# x, y = (torch.tensor(np.load('training_data/{}.npy'.format(pth)))
#         for pth in ['FFBP128', 'Phantom', ])
# x_test, y_test = x[3995:], y[3995:, 0]
quarts = []
for q in range(4):
    x_test_rot = torch.tensor(rot(x_test, q*90, axes=(2, 3))[:, :, :256, :256]).cuda()
#     out = model.predict(x_test_rot).cpu()
    with torch.no_grad():
        one_hot_round, img = prev_model.predict(x_test_rot)
        x_input = torch.cat((x_test.cuda(), img, one_hot_round), 1)
        rot = model.predict(x_input)
    quarts.append(rot(out, -90*q, axes=(1, 2)))
b = x_test.shape[0]
out = np.zeros([b, 512, 512])
out[:, :256, :256] = quarts[0]
out[:, :256, 256:] = quarts[1]
out[:, 256:, 256:] = quarts[2]
out[:, 256:, :256] = quarts[3]
np.save('validation_data/prediction.npy', out)

