import numpy as np
fruits = np.load('fruits_300.npy')
fruits_2d = fruits.reshape(-1, 100*100)

from sklearn.cluster import KMeans
km = KMeans(n_clusters = 3, random_state = 42)
km.fit(fruits_2d)

print(km.labels_)
print(np.unique(km.labels_, return_counts=True))

import matplotlib.pyplot as plt


inertia = []
for k in range(2,14):
    km = KMeans(n_clusters=k, n_init='auto', random_state=42)
    km.fit(fruits_2d)
    inertia.append(km.inertia_)
plt.plot(range(2,14), inertia)
plt.xlabel('k')
plt.ylabel('inertia')
plt.show()
