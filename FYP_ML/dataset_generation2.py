import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Example1: make_classification()生成二分类数据集
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1000, n_features=20, n_redundant=0, n_clusters_per_class=1, n_informative=1,
                           n_classes=2, random_state=20)

# scatter plot, dots colored by class value
df = pd.DataFrame(dict(x=X[:, 14], y=X[:,10], label=y))
colors = {0: 'red', 1: 'blue'}
fig, ax = plt.subplots()
grouped = df.groupby('label')
for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
plt.show()
print(X.shape, y.shape)