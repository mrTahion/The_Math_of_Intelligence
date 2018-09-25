import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

# Data file path
data_file_path= 'input/glass.csv'

# reading data csv
data = pd.read_csv(data_file_path)


columns_drop= ['Type']
# pull data into type (y) and features (X)
y_train = data.Type
X_train = data.drop(columns_drop, axis=1)

# data prepdor PCA
PCAdata = (X_train - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)

# Getting covariance matrix
cov = np.matmul(PCAdata.transpose(), PCAdata) / (PCAdata.shape[0] - 1)
# Get eigen values and axis
eig, axes = np.linalg.eig(cov)

# calculete components for 2d - 2 largest eigen values
pc2d = np.matmul(PCAdata, axes[:, 0:2])
pc1 = np.array(pc2d[:, 0].transpose())
pc2 = np.array(pc2d[:, 1].transpose())

# Plot
plt.scatter(pc1, pc2, c=y_train)
plt.show()