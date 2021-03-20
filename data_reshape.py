import torch
import numpy as np

for pth in ['Phantom', 'FFBP128', 'Sinogram']:
    for batch in range(1, 5):
        data_batch = np.load('training_data/batches/{}_batch{}.npy'.format(pth, batch))
        if batch==1:
            data = data_batch
        else:
            data = np.concatenate((data, data_batch), axis=0)
    data = np.expand_dims(data, 1)
    np.save('training_data/' + pth + '.npy', data)


