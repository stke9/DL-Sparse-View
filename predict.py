import torch
from networks import *
import numpy as np
import matplotlib.pyplot as plt


model = DnCNN_OHE(in_ch=1, out_ch=5, depth=18, ch=48)
model.load_state_dict(torch.load('18layer_48ch.pt'))
model.eval()
model.cuda()

# x_test = torch.tensor(np.load('training_data/Phantom_batch_first_10.npy')).unsqueeze(1).cuda()
x, y = (torch.tensor(np.load('training_data/{}.npy'.format(pth)))
        for pth in ['FFBP128', 'Phantom', ])
x_test, y_test = x[3995:], y[3995:, 0]

x_pred = model.predict(x_test.cuda()).cpu()
# loss = loss_func(imgs, y_test[:, 0])
# print(loss**0.5)
fig, axs = plt.subplots(3, 3)
for i, (x, y) in enumerate(zip(x_pred[:3], y_test)):
    axs[i, 0].imshow(x)
    axs[i, 1].imshow(y)
    axs[i, 2].imshow((x - y)**2)


err = (x_pred - y_test)**2
for ph in [0, 0.194, 0.233]:
    mask1 = torch.logical_and(x_pred == ph, y_test != ph)
    mask2 = torch.logical_and(x_pred != ph, y_test == ph)
    print('model assigns phase incorrectly', len(x_pred[mask1]), ' resulting error: ', (err[~mask1].mean()**0.5).item())
    print('model fails to assign phase', len(x_pred[mask2]), ' resulting error: ', (err[~mask2].mean()**0.5).item())

err_grad = err[(y_test == 0) | (y_test == 0.194) | (y_test == 0.233)]
print('without edge error:', (err_grad.mean()**0.5).item())
# mask2 = torch.logical_and(x_pred != ph, y_test == ph)

for e in range(12):
    err_grad = err[err < 10**-e]
    print('without error from top', 100-len(err_grad)*100/len(err.reshape(-1)), '% of pixel errors:',  (err_grad.mean()**0.5).item())

plt.close('all')
i=5
f, a, b = np.where(err>0.0002)
for im in [x_test, err, y_test, x_pred]:
    plt.figure()
    plt.imshow(im[f[i], a[i]-20:a[i]+20, b[i]-20:b[i]+20])