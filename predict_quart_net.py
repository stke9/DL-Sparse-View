import torch
from networks import *
import numpy as np
from scipy.ndimage import rotate as rot
import matplotlib.pyplot as plt


model = DnCNN_OHE_res(in_ch=1, out_ch=5, depth=18, ch=128)
model.load_state_dict(torch.load('12layer_128ch_quart_img.pt'))
model.eval()
model.cuda()

x_test = torch.tensor(np.load('validation_data/FBP128_validation.npy')).unsqueeze(1)
x, y = (torch.tensor(np.load('training_data/{}.npy'.format(pth)))
        for pth in ['FFBP128', 'Phantom', ])
x_test, y_test = x[3995:], y[3995:, 0]
quarts = []
for q in range(4):
    out = model.predict(torch.rot90(x_test, q, dims=(2, 3))[:, :, :256, :256].cuda()).cpu()
    quarts.append(torch.rot90(out, 4-q, dims=(1, 2)))
b = x_test.shape[0]
out = np.zeros([b, 512, 512])
out[:, :256, :256] = quarts[0]
out[:, :256, 256:] = quarts[1]
out[:, 256:, 256:] = quarts[2]
out[:, 256:, :256] = quarts[3]
np.save('validation_data/predictions.npy', out)
# # loss = loss_func(imgs, y_test[:, 0])
# # print(loss**0.5)
# fig, axs = plt.subplots(3, 3)
# for i, (x, y) in enumerate(zip(x_pred[:3], y_test)):
#     axs[i, 0].imshow(x)
#     axs[i, 1].imshow(y)
#     axs[i, 2].imshow((x - y)**2)
#
# #
# err = (x_pred - y_test)**2
# for ph in [0, 0.194, 0.233]:
#     mask1 = torch.logical_and(x_pred == ph, y_test != ph)
#     mask2 = torch.logical_and(x_pred != ph, y_test == ph)
#     print('model assigns phase incorrectly', len(x_pred[mask1]), ' resulting error: ', (err[~mask1].mean()**0.5).item())
#     print('model fails to assign phase', len(x_pred[mask2]), ' resulting error: ', (err[~mask2].mean()**0.5).item())
#
# err_grad = err[(y_test == 0) | (y_test == 0.194) | (y_test == 0.233)]
# print('without edge error:', (err_grad.mean()**0.5).item())
# # mask2 = torch.logical_and(x_pred != ph, y_test == ph)
#
# for e in range(12):
#     err_grad = err[err < 10**-e]
#     print('without error from top', 100-len(err_grad)*100/len(err.reshape(-1)), '% of pixel errors:',  (err_grad.mean()**0.5).item())
#
# plt.close('all')
# i=5
# f, a, b = np.where(err>0.0002)
# for im in [x_test, err, y_test, x_pred]:
#     plt.figure()
#     plt.imshow(im[f[i], a[i]-20:a[i]+20, b[i]-20:b[i]+20])