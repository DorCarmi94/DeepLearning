# DeepLearning
Reconstructing and classifying MNIST images
Reconstructing and predicting the S&P500 index




MNIST. MNIST is a well-known dataset that can be downloaded from http://yann.lecun.
com/exdb/mnist/. Most existing deep learning frameworks have built-in methods for downloading
and pre-processing MNIST. For instance, torchvision of pyTorch, see https://pytorch.org/
vision/stable/datasets.html#mnist. MNIST images are not a natural time series, however,
they can be treated as a time series in the following way. Specifically, when considered as a
sequence {xt}, each xt
is a different pixel of a given MNIST image.

S&P500 stock prices. We consider the stock prices of S&P500 over the years 2014-2017. The
data is available via http://www.kaggle.com 
