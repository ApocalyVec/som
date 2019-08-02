# Self Organizing Map

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Credit_Card_Applications.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X)

# Training the SOM
from minisom import MiniSom
som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5)
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 100)

# Visualizing the results
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's']
colors = ['r', 'g']
for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()

# Finding the frauds
mappings = som.win_map(X)
# TODO: because self-organizing map organizes differently everythime it is created, the indices of frauds in mappings needs to be changed accordingly
frauds = np.concatenate((mappings[(3, 2)], mappings[(8, 9)]), axis=0)
frauds = sc.inverse_transform(frauds)

# create the matrix of features
customers = dataset.iloc[:, 1:].values

# create the dependent vairable
is_fraud = np.zeros((len(dataset)))
for i in range(len(dataset)):
    # dataset.iloc[i, 0] gets the customer id for customer number i
    if dataset.iloc[i, 0] in frauds:
        is_fraud[i] = 1

# ANN
# Feature Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
customers = sc.fit_transform(customers)

# Part 2 - building the ANN
from keras.models import Sequential
from keras.layers import Dense

# Initializing the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
# As a general practice: the number of neurons in each layer is ave(input_dim + output_dim), in this case,
# number of neurons = (11 + 1)/2 = 6
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=15))

# Adding the second hidden layer
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))

# Adding the output hidden layer
# units = 1 for output_dim = 1
classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

# Compiling the ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fitting the ANN to the Training set - the actual training
classifier.fit(customers, is_fraud, batch_size=1, epochs=2)

# Predict the probabilities of frauds
y_pred = classifier.predict(customers)

# add the customer id column
y_pred = np.concatenate((dataset.iloc[:, 0:1].values, y_pred), axis=1)  # axis = 1 for horizontal concatenation
y_pred = y_pred[y_pred[:, 1].argsort()]