# sequence to sequcne with numpy, cupy
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import time
from Base.config import *
from Base.fucntions import sigmoid, MSE
from Base.optim import Adam

if GPU:
    import numpy


class TimeFC:
    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None

    def forward(self, x):
        W, b = self.params
        N, T, D = x.shape

        reshaped_x = x.reshape(N * T, -1)
        y = np.dot(reshaped_x, W) + b

        self.x = x
        y = y.reshape(N, T, -1)
        return y

    def backward(self, dy):
        W, b = self.params
        x = self.x
        N, T, D = x.shape

        dy = dy.reshape(N * T, -1)
        reshaped_x = x.reshape(N * T, -1)

        db = np.sum(dy, axis=0)
        dx = np.dot(dy, W.T)
        dW = np.dot(reshaped_x.T, dy)

        dx = dx.reshape(*x.shape)

        self.grads[0][...] = dW
        self.grads[1][...] = db

        return dx


class TimeMSE:
    def __init__(self):
        self.params, self.grads = [], []
        self.cache = None

    def forward(self, xs, ts):
        N, T, V = xs.shape
        xs = xs.reshape(N * T, V)
        ts = ts.reshape(N * T, 1)

        loss = MSE(xs, ts)
        self.cache = (ts, xs, (N, T, V))

        return loss

    def backward(self, dy=1):
        ts, xs, (N, T, V) = self.cache
        dx = dy * (xs - ts) / (N)
        dx = dx.reshape(N, T, V)
        return dx


class LSTM:
    def __init__(self, Wx, Wh, b):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.cache = None

    def forward(self, x, h_prev, c_prev):
        # Affine transformation (Wx[f, g, i, o], Wh[f, g, i, o], b[f, g, i, o])
        Wx, Wh, b = self.params
        N, H = h_prev.shape

        A = np.dot(x, Wx) + np.dot(h_prev, Wh) + b

        f = A[:, :H]
        g = A[:, H : 2 * H]
        i = A[:, 2 * H : 3 * H]
        o = A[:, 3 * H :]

        f = sigmoid(f)
        g = np.tanh(g)
        i = sigmoid(i)
        o = sigmoid(o)

        c_next = f * c_prev + g * i
        h_next = o * np.tanh(c_next)

        self.cache = (x, h_prev, c_prev, i, f, g, o, c_next)
        return h_next, c_next

    def backward(self, dh_next, dc_next):
        Wx, Wh, b = self.params
        x, h_prev, c_prev, i, f, g, o, c_next = self.cache

        tanh_c_next = np.tanh(c_next)

        ds = dc_next + (dh_next * o) * (1 - tanh_c_next ** 2)

        dc_prev = ds * f

        di = ds * g
        df = ds * c_prev
        do = dh_next * tanh_c_next
        dg = ds * i

        di *= i * (1 - i)
        df *= f * (1 - f)
        do *= o * (1 - o)
        dg *= 1 - g ** 2

        dA = np.hstack((df, dg, di, do))

        dWh = np.dot(h_prev.T, dA)
        dWx = np.dot(x.T, dA)
        db = dA.sum(axis=0)

        self.grads[0][...] = dWx
        self.grads[1][...] = dWh
        self.grads[2][...] = db

        dx = np.dot(dA, Wx.T)
        dh_prev = np.dot(dA, Wh.T)

        return dx, dh_prev, dc_prev


class TimeLSTM:
    def __init__(self, Wx, Wh, b, stateful=False, return_sequences=True):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.layers = None  # for LSTM layer
        self.h, self.c = None, None
        self.dh = None
        self.stateful = stateful
        self.return_sequences = return_sequences

    def set_state(self, h, c=None):
        self.h, self.c = h, c

    def reset_state(self):
        self.h, self.c = None, None

    def forward(self, xs):
        Wx, Wh, b = self.params
        N, T, D = xs.shape  # mini-batch, time length, Dimension
        self.T = T
        H = Wh.shape[0]  # Wh (H, 4H) H: hidden size

        self.layers = []  # for stacking LSTM layer (horizontal)
        hs = np.empty((N, T, H), dtype="f")  # for save (h0 ... ht)

        # if not stateful, initialize h and c
        if not self.stateful or self.h is None:
            self.h = np.zeros((N, H), dtype="f")
        if not self.stateful or self.c is None:
            self.c = np.zeros((N, H), dtype="f")

        for t in range(T):
            layer = LSTM(*self.params)
            self.h, self.c = layer.forward(xs[:, t, :], self.h, self.c)

            hs[:, t, :] = self.h
            self.layers.append(layer)

        if not self.return_sequences:
            hs = hs[:, -1, :]  # return last states

        return hs

    def backward(self, dhs):
        Wx, Wh, b = self.params
        N, T, H = dhs.shape
        D = Wx.shape[0]

        dxs = np.empty((N, T, D), dtype="f")
        dh, dc = 0, 0

        grads = [0, 0, 0]  # dWx, dWh, db
        for t in reversed(range(T)):  # BPTT
            layer = self.layers[t]
            dx, dh, dc = layer.backward(dhs[:, t, :] + dh, dc)
            # dx, dh, dc = layer.backward(dh, dc)
            dxs[:, t, :] = dx
            for i, grad in enumerate(layer.grads):
                grads[i] += grad

        for i, grad in enumerate(grads):
            self.grads[i][...] = grad

        self.dh = dh
        return dxs


class Encoder:
    def __init__(self, n_steps, n_hiddens, n_features):
        T, H, F = n_steps, n_hiddens, n_features
        rand = np.random.randn

        # initialize weights by Xavier
        lstm_Wx = (rand(F, 4 * H) / np.sqrt(F)).astype("f")
        lstm_Wh = (rand(H, 4 * H) / np.sqrt(H)).astype("f")
        lstm_b = np.zeros(4 * H).astype("f")

        self.layers = [TimeLSTM(lstm_Wx, lstm_Wh, lstm_b)]

        self.params, self.grads = [], []
        self.hs = None

        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, xs):
        """
        xs shape : (n_batchs, n_steps, n_features)
        """
        xs = np.array(xs)
        for layer in self.layers:
            output = layer.forward(xs)

        self.hs = output

        return output[:, -1, :]

    def backward(self, dy):
        dhs = np.zeros_like(self.hs)
        dhs[:, -1, :] = dy
        for layer in reversed(self.layers):
            dy = layer.backward(dhs)

        return dy


class Decoder:
    def __init__(self, n_steps, n_hiddens, n_features):
        T, H, F = n_steps, n_hiddens, n_features
        rand = np.random.randn

        # initialize weights by Xavier
        lstm_Wx = (rand(F, 4 * H) / np.sqrt(F)).astype("f")
        lstm_Wh = (rand(H, 4 * H) / np.sqrt(H)).astype("f")
        lstm_b = np.zeros(4 * H).astype("f")

        # fully connected layer by He
        fc_W1 = (rand(H, H) / np.sqrt(H / 2)).astype("f")
        fc_b1 = np.zeros(H).astype("f")

        fc_W2 = (rand(H, 1) / np.sqrt(H / 2)).astype("f")
        fc_b2 = np.zeros(1).astype("f")

        self.layers = [
            TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=True),
            TimeFC(fc_W1, fc_b1),
            TimeFC(fc_W2, fc_b2),
        ]

        self.params, self.grads = [], []

        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, xs, h):
        self.layers[0].set_state(h)

        xs = np.array(xs)
        for layer in self.layers:
            xs = layer.forward(xs)

        return xs

    def backward(self, dy):
        for layer in reversed(self.layers):
            dy = layer.backward(dy)

        dh = self.layers[0].dh

        return dh


class Seq2Seq:
    def __init__(self, time_size, hidden_size, feature_size):
        T, H, F = time_size, hidden_size, feature_size

        self.encoder = Encoder(T, H, F)
        self.decoder = Decoder(1, H, F)
        self.loss_layer = TimeMSE()

        self.params = self.encoder.params + self.decoder.params
        self.grads = self.encoder.grads + self.decoder.grads

    def reset_state(self):
        self.encoder.layers[0].reset_state()
        self.decoder.layers[0].reset_state()

    def predict(self, xs):
        xs = np.array(xs)
        h = self.encoder.forward(xs)

        dec_in = np.zeros((xs.shape[0], 1, xs.shape[2]))
        dec_out = self.decoder.forward(dec_in, h)

        return dec_out

    def forward(self, xs, ts):
        xs, ts = np.array(xs), np.array(ts)
        output = self.predict(xs)

        loss = self.loss_layer.forward(output, ts)
        return loss

    def backward(self, dy=1):
        dy = self.loss_layer.backward(dy)
        dy = self.decoder.backward(dy)
        dy = self.encoder.backward(dy)

        return dy

    def fit(self, train_X, train_y, learning_rate=0.01, epochs=10, batch_size=32, verbose=0):
        optimizer = Adam(learning_rate)

        data_size = train_X.shape[0]
        max_iters = data_size // batch_size

        for epoch in range(1, epochs + 1):

            # shuffle data
            if GPU:
                idx = numpy.random.permutation(numpy.arange(data_size))
            else:
                idx = np.random.permutation(np.arange(data_size))
            train_X = train_X[idx]
            train_y = train_y[idx]

            epcoh_loss = 0
            start_time = time.time()

            for iter in range(max_iters):
                batch_x = train_X[iter * batch_size : (iter + 1) * batch_size]
                batch_y = train_y[iter * batch_size : (iter + 1) * batch_size]

                loss = self.forward(batch_x, batch_y)
                self.backward()
                params, grads = self.params, self.grads
                optimizer.update(params, grads)
                epcoh_loss += loss

            avg_loss = epcoh_loss / max_iters

            if verbose:
                duration = start_time - time.time()
                print(f"epoch:{epoch}/{epochs}, time:{duration:.2f}[s], loss:{avg_loss:.5f}")


class PeekyDecoder:
    def __init__(self, n_steps, n_hiddens, n_features):
        T, H, F = n_steps, n_hiddens, n_features
        rand = np.random.randn

        # initialize weights by Xavier
        lstm_Wx = (rand(H + F, 4 * H) / np.sqrt(H + F)).astype("f")
        lstm_Wh = (rand(H, 4 * H) / np.sqrt(H)).astype("f")
        lstm_b = np.zeros(4 * H).astype("f")

        # fully connected layer by He
        fc_W1 = (rand(H + H, H) / np.sqrt((H + H) / 2)).astype("f")
        fc_b1 = np.zeros(H).astype("f")

        fc_W2 = (rand(H + H, 1) / np.sqrt((H + H) / 2)).astype("f")
        fc_b2 = np.zeros(1).astype("f")

        self.layers = [
            TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=True),
            TimeFC(fc_W1, fc_b1),
            TimeFC(fc_W2, fc_b2),
        ]

        self.params, self.grads = [], []

        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

        self.cache = None

    def forward(self, xs, h):
        N, T, D = xs.shape
        N, H = h.shape

        xs = np.array(xs)

        self.layers[0].set_state(h)

        hs = np.repeat(h, T, axis=0).reshape(N, T, H)

        xs = np.concatenate((hs, xs), axis=2)
        out = self.layers[0].forward(xs)
        out = np.concatenate((hs, out), axis=2)

        out = self.layers[1].forward(out)
        out = np.concatenate((hs, out), axis=2)

        out = self.layers[2].forward(out)
        self.cache = H
        return out

    def backward(self, dy):
        H = self.cache

        dy = self.layers[2].backward(dy)
        dy, dhs0 = dy[:, :, H:], dy[:, :, :H]
        dy = self.layers[1].backward(dy)
        dy, dhs1 = dy[:, :, H:], dy[:, :, :H]
        dy = self.layers[0].backward(dy)
        dy, dhs2 = dy[:, :, H:], dy[:, :, :H]

        dhs = dhs0 + dhs1 + dhs2
        dh = self.layers[0].dh + np.sum(dhs, axis=1)
        return dh


class PeekySeq2Seq(Seq2Seq):
    def __init__(self, time_size, hidden_size, feature_size):
        T, H, F = time_size, hidden_size, feature_size

        self.encoder = Encoder(T, H, F)
        self.decoder = PeekyDecoder(1, H, F)
        self.loss_layer = TimeMSE()

        self.params = self.encoder.params + self.decoder.params
        self.grads = self.encoder.grads + self.decoder.grads
