import numpy as np
samples_path = "./2024-04-03-20-22-57.npy"
samples = np.load(samples_path, allow_picle=True)
print(len(samples))
