# artificial-neural-network-ANN-regresssor-from-scratch-

Implemented an artificial neural network
(ANN) regresssor and its weights are learned using the
backpropagation algorithm. This model is tested on two datasets
which the instances of both datasets have 1-D inputs and 1-D out
puts.

First of all, in this model we use a class to create a Linear
Regression Model (without any hidden layer), and a Neural
Network Model with one hidden layer and desired number of
neurons. L is used to illustrate the number of layers used in the
model and n is used to initialize the number of neurons.
Second, for compiling and completing the model, an
activation function is used to select parameters like learning
rate and loss and after compiling, the model is ready to be
trained. In this step, the number of epochs to train and batch
size is provided and it returns a dictionary that has the loss and
prediction values.
