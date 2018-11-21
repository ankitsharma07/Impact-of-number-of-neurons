#Importing libraries
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

%matplotlib inline

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import SGD, Adam, RMSprop
from sklearn.preprocessing import *
from sklearn.cross_validation import *
from sklearn.metrics import *

#Importing dataset
TRAIN_FILE = 'data/train.csv'
TEST_FILE = 'data/test.csv'

#Data pre-processing
train_data = np.loadtxt(TRAIN_FILE, skiprows = 1, delimiter = ',', dtype = 'float')
X = train_data[:, 1:]

# Preprocess the data to make features fall between 0 and 1. Neural networks perform a lot better in this way.

X = X/255
raw_Y = train_data[:, 0].reshape(-1, 1)

X_test = np.loadtxt(TEST_FILE, skiprows = 1, delimiter = ',', dtype = 'float')

# Preprocess the data to make features fall between 0 and 1. Neural networks perform a lot better in this way.

X_test = X_test/255

X_train, X_cv, raw_Y_train, raw_Y_cv = train_test_split(X, raw_Y, test_size = 0.20)

# Converter to transform input into one hot encoding, i.e. [3] => [0, 0, 1, 0, 0, 0, 0, 0, 0, 0].
# Can use the np_utils from Keras instead.

Y_expander = OneHotEncoder().fit(raw_Y)
Y_train = Y_expander.transform(raw_Y_train).astype(int).toarray()
Y_cv = Y_expander.transform(raw_Y_cv).astype(int).toarray()

#Training the model

n_hiddens = [512, 256, 128, 64, 32, 16, 8, 4, 2, 1]
scores = []
for n_hidden in n_hiddens:
    # Build a simple neural network.
    model = Sequential()
    model.add(Dense(input_dim = X.shape[1], output_dim = n_hidden))
    model.add(Activation('tanh'))
    model.add(Dense(output_dim = 10))
    model.add(Activation('softmax'))
    sgd = SGD(lr=0.2, decay=1e-7, momentum=0.1, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer='sgd')

    model.fit(X_train, Y_train, nb_epoch = 10, batch_size = 10, show_accuracy = True, verbose = 1, validation_split = 0.05)
    Y_cv_pred = model.predict_classes(X_cv, batch_size = 10, verbose = 1)

    score = accuracy_score(raw_Y_cv, Y_cv_pred)
    scores.append(score)
    print('Using [%d] number of hidden neurons yields. Accuracy score: %.4f' % (n_hidden, score))
    print('')


# Plot the results for comparison

fig = plt.figure()
fig.suptitle('(Test validation score) against (Number of hidden neurons) on MNIST data', fontsize = 20)
fig.set_figwidth(17)
fig.set_figheight(8)
ax = fig.add_subplot(111)
ax.plot(n_hiddens, scores, '-o', markersize = 10, markerfacecolor = 'r')
ax.set_xlabel('Number of hidden neurons', fontsize = 14)
ax.set_ylabel('Accuracy score', fontsize = 14)


