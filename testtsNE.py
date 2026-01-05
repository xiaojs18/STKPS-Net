from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import os
import torch

a=torch.rand(5,2048)
b=torch.tensor([0,1,2,3,4])
X_tsne = TSNE(n_components=2,random_state=33).fit_transform(a)



plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=b,label="t-SNE")
plt.legend()
plt.show()

