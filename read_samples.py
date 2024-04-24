import numpy as np
S = np.load("samples.npy", allow_pickle=True)
for i in range(len(S)):
    s = S[i]
    for j in range(len(s)):
        if (len(s[j])) > 3:
            print("!")
