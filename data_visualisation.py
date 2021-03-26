import numpy as np
import matplotlib.pyplot as plt
from util import *

datasets = [np.load('training_data/{}.npy'.format(pth))
        for pth in ['Phantom', 'FFBP128', 'Sinogram']]



# plot some images
plot = False

if plot:
    nimgs = 2
    fig, ax = plt.subplots(2, nimgs)
    for i, data in enumerate(datasets):
        for j in range(nimgs):
            ax[i, j].imshow(data[j, 0], vmin=0.15)
    plt.tight_layout()

# baseline MSE
EC = True
if EC:
    phant = datasets[0]
    img = one_hot_y(torch.tensor(phant[:10]))
    fbp = prep_img(datasets[1])
    #
    plt.figure()
    diff = (((fbp - phant)**2)).mean(axis=(1, 2, 3))**0.5
plt.hist(diff)

