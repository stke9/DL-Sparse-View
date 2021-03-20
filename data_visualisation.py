import numpy as np
import matplotlib.pyplot as plt

datasets = [np.load('training_data/{}_batch1.npy'.format(pth))
        for pth in ['Phantom', 'FFBP128', 'Sinogram']]



# plot some images
plot = False

if plot:
    nimgs = 2
    fig, ax = plt.subplots(2, nimgs)
    for i, data in enumerate(datasets[:-1]):
        for j in range(nimgs):
            ax[i, j].imshow(data[j])
    plt.tight_layout()

# baseline MSE
phant = datasets[0]
fbp = datasets[1]

diff = (((fbp - phant)**2)**0.5).mean(axis=(1,2))
plt.hist(diff)

