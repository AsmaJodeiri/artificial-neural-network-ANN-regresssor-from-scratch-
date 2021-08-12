import numpy as np

class ANN():
    def __init__(self, L=2, n=10):
        '''
        L: number of layers (input layer is not counted)
        n: number of neurons if there L=2 (which means there is a hidden layer)
        Our default model would be a NN model without 1 hidden layer, but user can change it,
        simply by just setting both values of L and n to 0 to use Linear Regression model
        '''
        if L != 1 and L != 2:
            raise Exception('Our model only supports Linear Regression (L=1)' \
             ' and NN with one hidden layer (L=2)')
        self.L = L
        self.n = n
        self.trained = False

    def compile(self, normalize_input=True, activation='sigmoid', learning_rate=0.3, loss_type='squared', beta=0.9):
        '''
        This method should be executed before training to specify loss function, activation function, ...
        '''

        # Check activation function
        self.normalize_input = normalize_input
        if activation != 'sigmoid' and activation != 'relu' and activation != 'tanh':
            raise Exception('Our model only supports these activations:\'sigmoid\', \'relu\', \'tanh\'')
        self.activation = activation
        self.learning_rate = learning_rate
        self.loss_type = loss_type
        self.beta = beta

        # Initializing the momentum values
        self.v_dw1 = 0
        self.v_dw2 = 0
        self.v_db1 = 0
        self.v_db2 = 0

    def normalize(self, y):
        x_normalized = (y - self.mean) / self.std
        return x_normalized

    def denormalize(self, y_normalized):
        y = y_normalized * self.std + self.mean
        return y

    def preprocess_x(self, x):
        # Reshape x_input to adapt our model notation
        x = np.array(x)
        if len(x.shape) == 1:
            x = np.expand_dims(x, axis=1)
        x = x.T         # T method will keep the 1 in shape if x_input has only one column, To clarify that
                        # as an example np.transpose changes shape from (20, 1) to (20, ) but T method changes
                        # it to (1, 20)

        return x

    def preprocess_y(self, y):
        # Normalize data in case of being required
        if self.normalize_input:
            y = self.normalize(y)
        return y

    def initialize_weights(self):
        resize_factor = 0.01    # Initial weight values should be small
        # Initialize the weight and bias matrices
        # Our input value is always a number so it has a shape of (1, 1)
        # We use dictionaries for weights and biases so we can start from layer 1 instead of 0 (if it was a list)
        self.biases = {}        # biases can be initialized to zero
        self.weights = {}       # List of weight of all layers
        if self.L == 1:
            b = np.zeros((1, 1))
            w = np.random.randn(1, 1) * resize_factor
            self.biases[1] = b
            self.weights[1] = w
        elif self.L == 2:
            b1 = np.zeros((self.n, 1))
            w1 = np.random.randn(self.n, 1) * resize_factor
            b2 = np.zeros((1, 1))
            w2 = np.random.randn(1, self.n) * resize_factor
            self.biases[1] = b1
            self.weights[1] = w1
            self.biases[2] = b2
            self.weights[2] = w2

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_prime(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def relu(self, z):
        return np.maximum(z, 0)

    def relu_prime(self, z):
        z[z >= 0] = 1
        z[z < 0] = 0
        return z

    def tanh(self, z):
        return np.tanh(z)

    def tanh_prime(self, z):
        return 1 - (np.square(np.tanh(z)))

    def compute_loss(self, y_pred, y):
        m = len(y)
        y_pred = y_pred.reshape(y.shape)
        error = np.abs(y_pred - y)
        # loss: Mean Squared Difference of Loss Values
        # mean_loss: Mean of Loss Values
        # std: Standard Deviation of Loss Values
        if self.loss_type == 'squared':
            loss = (1 / (2 * m)) * np.sum(np.square(error))
        elif self.loss_type == 'absolute':
            loss = (1 / m) * np.absolute(error)

        # loss_mean = np.mean(error)
        # loss_std = np.std(error)

        return loss

    def forward_propagation(self, x, y=None):
        # Add an attribute to the model for storing hidden layer values.
        # It is a dictionary that the keys are the index of neuron
        self.a1 = {}

        z = {}
        a = {}
        a[0] = np.array(x)         # a0 is indeed x (input) itself
        # FORWARD PROPAGATION
        for l in range(1, self.L+1):
            # print('l:', l)
            z_l = np.dot(self.weights[l], a[l-1]) + self.biases[l]
            if l == self.L:
                a_l = z_l
            else:
                if self.activation == 'sigmoid':
                    a_l = self.sigmoid(z_l)
                elif self.activation == 'relu':
                    a_l = self.relu(z_l)
                elif self.activation == 'tanh':
                    a_l = self.tanh(z_l)

            z[l] = z_l
            a[l] = a_l

            # Store the values of hidden layers
            # print('shape of a: ', a[1].shape)
            if self.L == 2:
                for i_neuron in range(a[1].shape[0]):
                    self.a1[i_neuron] = a[1][i_neuron, :]


        # COMPUTE THE LOSS OF PRESENT EPOCH
        y_pred = a[self.L]
        if y is None:
            # This is used for just prediction purposes
            return y_pred
        else:
            # This is for training purposes that need cache and displays loss values during training
            loss = self.compute_loss(y_pred, y)
            cache = {
                'z': z,
                'a': a,
            }
        return y_pred, loss, cache

    def backward_propagation(self, x, y, y_pred, cache):
        # We handle L=1 and L=2 separately in backward propagation
        m = len(y)
        if self.L == 1:
            dz = y_pred - y
            dw = (1 / m) * np.dot(x, np.transpose(dz))
            db = (1 / m) * np.sum(dz, axis=1, keepdims=True)

            # Compute momentums (only v_dw1 and v_db1 in the model without hidden layer)
            self.v_dw1 = self.beta * self.v_dw1 + (1 - self.beta) * dw
            self.v_db1 = self.beta * self.v_db1 + (1 - self.beta) * db

            # Update weights and biases
            self.weights[1] -= self.learning_rate * self.v_dw1
            self.biases[1] -= self.learning_rate * self.v_db1
        elif self.L == 2:
            # Retrieve the weights
            w1 = self.weights[1]
            w2 = self.weights[2]
            # We saved z1 in cache so we can use it for back propagation
            z = cache['z']
            z1 = z[1]
            a = cache['a']
            a1 = a[1]
            dz2 = y_pred - y
            dw2 = (1 / m) * np.dot(dz2, np.transpose(a1))
            db2 = (1 / m) * np.sum(dz2, axis=1, keepdims=True)
            # Compute dz1 due to the assigned activation function (we need to use its derivative here)
            if self.activation == 'sigmoid':
                dz1 = np.multiply(np.dot(np.transpose(w2), dz2), self.sigmoid_prime(z1))
            elif self.activation == 'relu':
                dz1 = np.multiply(np.dot(np.transpose(w2), dz2), self.relu_prime(z1))
            elif self.activation == 'tanh':
                dz1 = np.multiply(np.dot(np.transpose(w2), dz2), self.tanh_prime(z1))
            dw1 = (1 / m) * np.dot(dz1, np.transpose(x))
            db1 = (1 / m) * np.sum(dz1, axis=1, keepdims=True)

            # Compute momentums
            self.v_dw1 = self.beta * self.v_dw1 + (1 - self.beta) * dw1
            self.v_dw2 = self.beta * self.v_dw2 + (1 - self.beta) * dw2
            self.v_db1 = self.beta * self.v_db1 + (1 - self.beta) * db1
            self.v_db2 = self.beta * self.v_db2 + (1 - self.beta) * db2


            # Update weights and biases
            self.weights[1] -= self.learning_rate * self.v_dw1
            self.biases[1] -= self.learning_rate * self.v_db1
            self.weights[2] -= self.learning_rate * self.v_dw2
            self.biases[2] -= self.learning_rate * self.v_db2

    def predict(self, x):
        if self.trained:
            x = self.preprocess_x(x)
            y_pred = self.forward_propagation(x)
            y_pred = np.transpose(y_pred)

            # Denormalize data in case of being required
            if self.normalize_input:
                y_pred = self.denormalize(y_pred)
            return y_pred
        else:
            print('You should train the model first and the execute the predict method.')

    def train_model(self, x_train, y_train, num_epochs=15000, batch_size=None, stop_loss=None):
        # We use train data to calculate mean and std in training, so they will not change for test data
        self.mean = np.mean(y_train)      # We need mean value to denormalize, so we save it as an attribute
        self.std = np.std(y_train)        # We need mean std to denormalize, so we save it as an attribute

        x = self.preprocess_x(x_train)
        y = self.preprocess_y(y_train)

        # If batch size is not specified, we set it to training size corresponding to batch learning
        if batch_size == None:
            batch_size = x.shape[1]
            # print(x.shape[1])
            # print('batch size:', batch_size)


        # Initialize the weights
        self.initialize_weights()

        # We create a list to save the history of our train at each epoch
        self.history = {
            'loss': [],
            'y_pred': [],
        }

        # We train our model for "num_epochs" times
        for epoch in range(num_epochs):
            # print('batch size:', batch_size)

            # Compute the batch gradient descent and stochastic gradient descent separately
            if batch_size == x.shape[1]:
                # Forward propagation
                y_pred, loss, cache = self.forward_propagation(x, y)
                self.history['y_pred'].append(y_pred)
                self.history['loss'].append(loss)
                print('Loss at epoch #{}: {:.4f}'.format(epoch, loss))
                if (stop_loss is not None) and (loss < stop_loss):
                    print('\nReached the minimum of value that has been designated.')
                    break

                # Back propagation (update weights)
                self.backward_propagation(x, y, y_pred, cache)

            else:
                num_batches = int(np.ceil(x.shape[1] / batch_size))
                for t in range(num_batches):
                    # Forward propagation
                    x_batch = x[:, t*batch_size:(t+1)*batch_size]
                    y_batch = y[t*batch_size:(t+1)*batch_size]
                    y_pred, loss, cache = self.forward_propagation(x_batch, y_batch)
                    self.history['y_pred'].append(y_pred)
                    self.history['loss'].append(loss)
                    print('Loss at epoch #{}: {:.4f}'.format(epoch, loss))
                    if (stop_loss is not None) and (loss < stop_loss):
                        print('\nReached the minimum of value that has been designated.')
                        break

                    # Back propagation (update weights)
                    self.backward_propagation(x_batch, y_batch, y_pred, cache)

                # Shuffle the data to avoid getting stuck in bad batches
                shuffled_indices = np.arange(x.shape[1])
                np.random.shuffle(shuffled_indices)
                x = x[:, shuffled_indices]
                y = y[shuffled_indices]

        # At the end of training, set self.trained to True and return history of training
        self.trained = True
        return self.history




####################################################
############## TRAIN & TEST THE MODEL ##############
####################################################
import matplotlib.pyplot as plt

# PLOT ACTUAL VALUES VS. PREDICTIONS
def plot_data(x, y, model, data_name):
    a = np.min(x)
    b = np.max(x)
    interval = np.linspace(a, b, 100).reshape(-1, 1)        # Reshape for adaption to the model
    # pred_scattered = model.predict(x)
    pred = model.predict(interval)
    plt.scatter(x, y, color='blue', label='Actual Data')
    # plt.scatter(x, pred_scattered, s=100, color='green', label='Predictions', marker='x', linewidths=3)
    plt.plot(interval, pred, color='red', linewidth=2, label='Prediction Plot')
    plt.legend()
    plt.title('Actual Data & Predictions of: "{}"'.format(data_name))
    plt.savefig('{}.png'.format(data_name))
    plt.show()

def log_loss(y_pred, y, log_file):
    m = len(y)
    # print(m)
    error = np.abs(y_pred.reshape(y.shape) - y)

    loss_rmsd = (1 / (2 * m)) * np.sum(np.square(error))        # Mean Squared Difference of Loss Values
    loss_mean = np.mean(error)                                  # Mean of Loss Values
    loss_std = np.std(error)                                    # Standard Deviation of Loss Values

    # print('RMSD: {} | MEAN: {} | STD {}'.format(loss_rmsd, loss_mean, loss_std))
    file = open(log_file, 'w')
    file.write('{:<10.3f},{:<10.3f},{:<10.3f}\n'.format(loss_rmsd, loss_mean, loss_std))
    file.close()



## FIRST DATASET
train_data = np.loadtxt('train1.txt')
test_data = np.loadtxt('test1.txt')


x_train, y_train = train_data[:, 0], train_data[:, 1]
x_test, y_test = test_data[:, 0], test_data[:, 1]


ann = ANN(L=2, n=10)

# TRAIN OVER FIRST DATASET (TRAIN & TEST)
ann.compile(learning_rate=0.3)
history = ann.train_model(x_train, y_train, num_epochs=20000)

# PLOT LOSS VALUES
plt.scatter(range(len(history['loss'])), history['loss'], s=3, edgecolors='blue')
# plt.legend()
plt.grid()
# plt.title('Loss values of models with momentum vs without momentum'.title())
plt.title('Loss Values'.format())
plt.savefig('loss_train1.png')
plt.show()


train_log = 'train_loss1.txt'
test_log = 'test_loss1.txt'

train_preds = ann.predict(x_train)
test_preds = ann.predict(x_test)

log_loss(train_preds, y_train, train_log)
log_loss(test_preds, y_test, test_log)

plot_data(x_train, y_train, ann, 'train1')
plot_data(x_test, y_test, ann, 'test1')


## SECOND DATASET
train_data = np.loadtxt('train2.txt')
test_data = np.loadtxt('test2.txt')


x_train, y_train = train_data[:, 0], train_data[:, 1]
x_test, y_test = test_data[:, 0], test_data[:, 1]


ann2 = ANN(L=2, n=25)

# TRAIN OVER FIRST DATASET (TRAIN & TEST)
ann2.compile(learning_rate=0.01)
history = ann2.train_model(x_train, y_train, num_epochs=200000)

# PLOT LOSS VALUES
plt.scatter(range(len(history['loss'])), history['loss'], s=3, edgecolors='blue')
# plt.legend()
plt.grid()
# plt.title('Loss values of models with momentum vs without momentum'.title())
plt.title('Loss Values'.format())
plt.savefig('loss_train2.png')
plt.show()


train_log = 'train_loss2.txt'
test_log = 'test_loss2.txt'

train_preds = ann2.predict(x_train)
test_preds = ann2.predict(x_test)

log_loss(train_preds, y_train, train_log)
log_loss(test_preds, y_test, test_log)

plot_data(x_train, y_train, ann2, 'train2')
plot_data(x_test, y_test, ann2, 'test2')