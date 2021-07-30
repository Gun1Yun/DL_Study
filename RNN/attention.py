import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import time
from seq2seq_np import *
from Base.config import *
from Base.fucntions import *
from Base.optim import Adam

if GPU:
    import numpy


class Softmax:
    def __init__(self):
        self.params, self.grads = [], []
        self.out = None

    def forward(self, x):
        self.out = softmax(x)
        return self.out

    def backward(self, dout):
        dx = self.out * dout
        sumdx = np.sum(dx, axis=1, keepdims=True)
        dx -= self.out * sumdx
        return dx


# get Attention score
class AttentionWeight:
    def __init__(self):
        self.params, self.grads = [], []
        self.softmax = Softmax()
        self.cache = None

    def forward(self, h, hs):
        """
        h : hidden state of decoder LSTM
            (N, H)  = (n_batchs, n_features)
        hs : Encoder's hidden state sequence
            (N, T, H) = (n_batchs, n_times, n_hiddens)
        """
        N, T, H = hs.shape

        # reapeat hidden state T times
        hr = h.reshape(N, 1, H).repeat(T, axis=1)  # (N, T, H)
        score = hs * hr  # (N, T, H)
        score = np.sum(score, axis=2)  # (N, T)
        attention_weight = self.softmax.forward(score)

        self.cache = (hs, hr)

        return attention_weight

    def backward(self, da):
        hs, hr = self.cache
        N, T, H = hs.shape

        ds = self.softmax.backward(da)
        ds = ds.reshape(N, T, 1).repeat(H, axis=2)
        dhs = ds * hr
        dhr = ds * hs
        dh = np.sum(dhr, axis=1)

        return dhs, dh


class WeightSum:
    def __init__(self):
        self.params, self.grads = [], []
        self.cache = None

    def forward(self, hs, a):
        N, T, H = hs.shape

        ar = a.reshape(N, T, 1).repeat(H, axis=2)
        weights = hs * ar
        weights_sum = np.sum(weights, axis=1)

        self.cache = (hs, ar)
        return weights_sum

    def backward(self, dws):
        hs, ar = self.cache
        N, T, H = hs.shape

        dw = dws.reshape(N, 1, H).repeat(T, axis=1)
        dhs = dw * ar
        dar = dw * hs
        da = np.sum(dar, axis=2)

        return dhs, da


class Attention:
    def __init__(self):
        self.params, self.grads = [], []
        self.attention_weight_layer = AttentionWeight()
        self.weight_sum_layer = WeightSum()
        self.attention_weight = None

    def forward(self, hs, h):
        a = self.attention_weight_layer.forward(h, hs)
        out = self.weight_sum_layer.forward(hs, a)
        self.attention_weight = a

        return out

    def backward(self, dout):
        dhs0, da = self.weight_sum_layer.backward(dout)
        dhs1, dh = self.attention_weight_layer.backward(da)

        dhs = dhs0 + dhs1
        return dhs, dh


class TimeAttention:
    def __init__(self):
        self.params, self.grads = [], []
        self.layers = None
        self.attention_weights = None

    def forward(self, hs_enc, hs_dec):
        N, T, H = hs_dec.shape
        out = np.empty_like(hs_dec)
        self.layers = []
        self.attention_weights = []

        for t in range(T):
            layer = Attention()
            out[:, t, :] = layer.forward(hs_enc, hs_dec[:, t, :])
            self.layers.append(layer)
            self.attention_weights.append(layer.attention_weight)

        return out

    def backward(self, dout):
        N, T, H = dout.shape
        dhs_enc = 0
        dhs_dec = np.empty_like(dout)

        for t in range(T):
            layer = self.layers[t]
            dhs, dh = layer.backward(dout[:, t, :])
            dhs_enc += dhs
            dhs_dec[:, t, :] = dh

        return dhs_enc, dhs_dec


class AttentionEncoder(Encoder):
    def forward(self, xs):
        xs = np.array(xs)

        for layer in self.layers:
            output = layer.forward(xs)
        self.hs = output

        return output

    def backward(self, dy):
        dout = self.layers[0].backward(dy)

        return dout


class AttentionDecoder:
    def __init__(self, n_steps, n_hiddens, n_features):
        T, H, F = n_steps, n_hiddens, n_features
        rand = np.random.randn

        # initialize weights by Xavier
        lstm_Wx = (rand(F, 4 * H) / np.sqrt(F)).astype("f")
        lstm_Wh = (rand(H, 4 * H) / np.sqrt(H)).astype("f")
        lstm_b = np.zeros(4 * H).astype("f")

        # fully connected layer by He
        fc_W1 = (rand(2 * H, H) / np.sqrt(2 * H / 2)).astype("f")
        fc_b1 = np.zeros(H).astype("f")

        fc_W2 = (rand(H, 1) / np.sqrt(H / 2)).astype("f")
        fc_b2 = np.zeros(1).astype("f")

        self.layers = [
            TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=True),
            TimeAttention(),
            TimeFC(fc_W1, fc_b1),
            TimeFC(fc_W2, fc_b2),
        ]

        self.params, self.grads = [], []

        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, xs, enc_hs):
        xs = np.array(xs)

        h = enc_hs[:, -1]
        self.layers[0].set_state(h)

        dec_hs = self.layers[0].forward(xs)
        a_out = self.layers[1].forward(enc_hs, dec_hs)
        out = np.concatenate((a_out, dec_hs), axis=2)
        score = self.layers[2].forward(out)
        score = self.layers[3].forward(score)

        return score

    def backward(self, dout):
        dout = self.layers[3].backward(dout)
        dout = self.layers[2].backward(dout)
        N, T, H2 = dout.shape  # because concat (attention_out, decoder_out)
        H = H2 // 2

        d_a, d_dec_hs0 = dout[:, :, :H], dout[:, :, H:]

        denc_hs, d_dec_hs1 = self.layers[1].backward(d_a)
        d_dec_hs = d_dec_hs0 + d_dec_hs1  # beacuse concat
        dout = self.layers[0].backward(d_dec_hs)
        dh = self.layers[0].dh

        denc_hs[:, -1] += dh

        return denc_hs


class AttentionSeq2Seq(Seq2Seq):
    def __init__(self, time_size, hidden_size, feature_size):
        T, H, F = time_size, hidden_size, feature_size
        self.encoder = AttentionEncoder(T, H, F)
        self.decoder = AttentionDecoder(1, H, F)
        self.loss_layer = TimeMSE()

        self.params = self.encoder.params + self.decoder.params
        self.grads = self.encoder.grads + self.decoder.grads
