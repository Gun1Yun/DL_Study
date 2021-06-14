from config import *
from layers import *

# CNN regression models
class CnnModelReg:
    def __init__(self, input_dim = (1, 24, 8), 
                 params={'filter_num':30, 'filter_size':5, 
                         'padding':0, 'stride':1},
                 hidden_size=100, output_size=1):
        filter_num = params['filter_num']
        filter_size = params['filter_size']
        padding = params['padding']
        stride = params['stride']

        # not square
        input_size = input_dim[1]
        input_size2 = input_dim[2]
        conv_output_size_h = (input_size - filter_size + 2*padding)/stride + 1
        conv_output_size_w = (input_size2 - filter_size + 2*padding)/stride + 1
        pool_output_size = int(filter_num* int(conv_output_size_h/2)* int(conv_output_size_w/2))

        self.params = {}
        rand = np.random.randn

        # He initialize
        self.params['cnn_W'] = rand(filter_num, input_dim[0], filter_size, filter_size) / np.sqrt(filter_num/2)
        self.params['cnn_b'] = np.zeros(filter_num)
        self.params['W1'] = rand(pool_output_size, hidden_size)/np.sqrt(pool_output_size/2)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = rand(hidden_size, output_size)/np.sqrt(hidden_size/2)
        self.params['b2'] = np.zeros(output_size)

        self.layers=[
                     Convolution(self.params['cnn_W'], self.params['cnn_b'], stride, padding),
                     ReLU(),
                     Pooling(pool_h=2, pool_w=2, stride=2),
                     FullyConnected(self.params['W1'], self.params['b1']),
                     ReLU(),
                     FullyConnected(self.params['W2'], self.params['b2']),
        ]
        self.loss_layer = ReluWithLoss()

        self.grads = []
        self.params = []
        for i in [0, 3, 5]:
            self.params += self.layers[i].params
            self.grads+=self.layers[i].grads

    def predict(self, x):
        x = np.array(x)
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def forward(self, x, t):
        x = np.array(x)
        t = np.array(t)
        x = self.predict(x)
        loss = self.loss_layer.forward(x, t)
        return loss

    def backward(self, dy=1):
        dy = self.loss_layer.backward(dy)

        for layer in reversed(self.layers):
            dy = layer.backward(dy)
        return dy
    
    def fit(self, train_X=None, train_y=None,learning_rate=0.01, epochs=10, batch_size=32, verbose=0):
        optimizer = Adam(learning_rate)

        data_size = train_X.shape[0]
        max_iters = data_size//batch_size

        for epoch in range(1, epochs+1):
            # shuffle train data
            idx = numpy.random.permutation(numpy.arange(data_size))
            x_data = train_X[idx]
            y_data = train_y[idx]

            epoch_loss = 0
            start_time = time.time()
            for iter in range(max_iters):
                batch_x = x_data[iter*batch_size:(iter+1)*batch_size]
                batch_y = y_data[iter*batch_size:(iter+1)*batch_size]

                loss = self.forward(batch_x, batch_y)
                self.backward()
                params, grads = self.params, self.grads
                optimizer.update(params, grads)

                epoch_loss += loss
            avg_loss = epoch_loss/max_iters

            if verbose:
                duration = start_time-time.time()
                print(f'epoch:{epoch}/{epochs}, 시간:{duration:.2f}[s], loss:{avg_loss:.5f}')



# LSTM Regression model
class LstmModelReg:
    def __init__(self, time_size, hidden_size, feature_size):
        T, H, F = time_size, hidden_size, feature_size
        H2 = 64
        rand = np.random.randn

        # weights (Xavier)
        lstm_Wx = (rand(F, 4*H)/ np.sqrt(F)).astype('f')
        lstm_Wh = (rand(H, 4*H)/ np.sqrt(H)).astype('f')
        lstm_b = np.zeros(4*H).astype('f')

        # He initialize
        fc_W1 = (rand(H, H2)/ np.sqrt(H/2)).astype('f')
        fc_b1 = np.zeros(H2).astype('f')

        fc_W2 = (rand(H2, 1)/ np.sqrt(H2/2)).astype('f')
        fc_b2 = np.zeros(1).astype('f')

        # layer
        self.layers = [
            TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=True),
            TimeFC(fc_W1, fc_b1),
            TimeFC(fc_W2, fc_b2)
        ]
        self.loss_layer = TimeMSE()

        self.params, self.grads = [], []

        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads


    def predict(self, xs):
        xs = np.array(xs)
        for layer in self.layers:
            xs = layer.forward(xs)
        return xs

    def forward(self, xs, ts):
        xs = np.array(xs)
        ts = np.array(ts)
        for layer in self.layers:
            xs = layer.forward(xs)
        loss = self.loss_layer.forward(xs, ts)
        return loss

    def backward(self, dy = 1):
        dy = self.loss_layer.backward(dy)
        for layer in reversed(self.layers):
            dy = layer.backward(dy)
        return dy

    def fit(self, train_X=None, train_y=None,learning_rate=0.01, epochs=10, batch_size=32, verbose=0):
        optimizer = Adam(learning_rate)

        data_size = train_X.shape[0]
        max_iters = data_size//batch_size

        for epoch in range(1, epochs+1):
            idx = numpy.random.permutation(numpy.arange(data_size))
            train_X = train_X[idx]
            train_y = train_y[idx]

            epoch_loss = 0
            start_time=time.time()
            
            for iter in range(max_iters):
                batch_x = train_X[iter*batch_size:(iter+1)*batch_size]
                batch_y = train_y[iter*batch_size:(iter+1)*batch_size]

                loss = self.forward(batch_x, batch_y)
                self.backward()
                params, grads = self.params, self.grads
                optimizer.update(params, grads)

                epoch_loss += loss
            avg_loss = epoch_loss/max_iters

            if verbose:
                duration = start_time-time.time()
                print(f'epoch:{epoch}/{epochs}, 시간:{duration:.2f}[s], loss:{avg_loss:.5f}')


    def reset_state(self):
        self.layers[0].reset_state()