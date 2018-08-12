import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from mnist import MNIST
images_path='train-images-idx3-ubyte'
labels_path='train-labels-idx1-ubyte'
n_features=28*28
n_classes=10
mndata=MNIST('Samples')
X,y=mndata.load_training()
X,y=shuffle_data(X,y,random_seed=RANDOM_SEED)
X_train,y_train=X[:500],y[:500]
X_test,y_test=X[500:],y[500:]
plot_digit(X,y,idx=1)
