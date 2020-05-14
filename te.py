import numpy as np


values = np.arange(0, 1, 1./5)

z_sample = np.random.uniform(-1, 1, size=(5,3))
for kdx, z in enumerate(z_sample):
    #z[idx] = values[kdx]
    print("z[",kdx,"]=",z)