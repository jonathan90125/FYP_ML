import numpy as np
from sklearn.datasets import make_classification

#design several effective parameters first(linear and onehot)
#use make regression
my_matrix, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
# np.savetxt("new.csv", X, delimiter=',')
# my_matrix = np.loadtxt(open("new.csv","rb"), delimiter=",", skiprows=0)

for x in range(100):
    my_matrix[x][1]=my_matrix[x][1]+1000
for x in range(100):
    my_matrix[x][2]=my_matrix[x][2]*20

np.savetxt("X.csv", my_matrix, delimiter=',')
np.savetxt("y.csv", y, delimiter=',')