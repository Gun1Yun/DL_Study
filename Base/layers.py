from config import *

# MSE for Loss
def MSE(y, t):
    return 0.5 * np.mean((y-t)**2)

# ReLU activation function
class ReLU:
    def __init__(self):
        self.mask = None
    
    def forward(self, x):
        self.mask = (x<=0)
        y = x.copy()
        y[self.mask] = 0
        return y

    def backward(self, dy):
        dy[self.mask] = 0
        dx = dy
        return dx

# Loss with ReLU
class ReluWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.activation = ReLU()
        self.cache = None

    def forward(self, x, t):
        N, V = x.shape      # batch, output
        
        x = x.reshape(N, V)
        t = t.reshape(N, V)
        x = self.activation.forward(x)

        loss = MSE(x, t)
        self.cache = (t, x, (N, V))
        return loss

    def backward(self, dy=1):
        t, x, (N, V) = self.cache
        dx = dy * (x-t) / N
        
        dx = self.activation.backward(dx)
        dx = dx.reshape(N, V)
        return dx


class FullyConnected:
    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None

    def forward(self, x):
        W, b = self.params
        self.x_shape = x.shape

        x = x.reshape(x.shape[0], -1)
        y = np.dot(x, W) + b        # y = X*W + b
        self.x = x

        return y

    def backward(self, dy):
        W, b = self.params
        x = self.x

        db = np.sum(dy, axis=0)
        dW = np.dot(x.T, dy)
        dx = np.dot(dy, W.T)

        self.grads[0][...] = dW
        self.grads[1][...] = db
        dx = dx.reshape(*self.x_shape)

        return dx

# CNN


def im2col(data, filter_h, filter_w, stride=1, padding=0):
    # flatten data to 2D array
    N, C, H, W = data.shape

    out_h = (H + 2*padding - filter_h)//stride + 1
    out_w = (W + 2*padding - filter_w)//stride + 1

    # padding for H, W
    img = np.pad(data,
                 [(0, 0), (0, 0), (padding, padding), (padding, padding)],
                 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)

    return col


def col2im(col, shape, filter_h, filter_w, stride=1, padding=0):
    # 2D for img data
    # shape = original data shape
    N, C, H, W = shape
    out_h = (H + 2*padding - filter_h)//stride + 1
    out_w = (W + 2*padding - filter_w)//stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h,
                      filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H+2*padding + stride - 1, W+2*padding+stride-1))

    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] = + col[:, :, y, x, :, :]

    return img[:, :, padding:H+padding, padding:W+padding]


class Convolution:
    def __init__(self, W, b, stride=1, padding=0):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.stride = stride
        self.padding = padding
        self.cache = None

    def forward(self, x):
        weight, b = self.params
        FN, FC, FH, FW = weight.shape
        N, C, H, W = x.shape

        out_h = (H + 2*self.padding - FH)//self.stride + 1
        out_w = (W + 2*self.padding - FW)//self.stride + 1

        col = im2col(x, FH, FW, self.stride, self.padding)
        col_W = weight.reshape(FN, -1).T

        y = np.dot(col, col_W) + b
        y = y.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        self.cache = (x, col, col_W)

        return y

    def backward(self, dy):
        W, b = self.params
        x, col, col_W = self.cache
        FN, C, FH, FW = W.shape

        dy = dy.transpose(0, 2, 3, 1).reshape(-1, FN)

        db = np.sum(dy, axis=0)
        dW = np.dot(col.T, dy)
        dW = dW.transpose(1, 0).reshape(FN, C, FH, FW)

        self.grads[0][...] = dW
        self.grads[1][...] = db

        dx = np.dot(dy, col_W.T)
        dx = col2im(dx, x.shape, FH, FW, self.stride, self.padding)

        return dx


class Pooling:
    def __init__(self, pool_h, pool_w, stride=1, padding=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.padding)
        col = col.reshape(-1, self.pool_h*self.pool_w)

        arg_max = np.argmax(col, axis=1)
        y = np.max(col, axis=1)
        y = y.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.cache = (x, arg_max)

        return y

    def backward(self, dy):
        x, arg_max = self.cache

        dy = dy.transpose(0, 2, 3, 1)      # N, C, H, W
        pool_size = self.pool_h*self.pool_w

        dmax = np.zeros((dy.size, pool_size))
        dmax[np.arange(arg_max.size), arg_max.flatten()] = dy.flatten()
        dmax = dmax.reshape(dy.shape + (pool_size, ))
        dcol = dmax.reshape(dmax.shape[0]*dmax.shape[1]*dmax.shape[2], -1)
        dx = col2im(dcol, x.shape,
                    self.pool_h,
                    self.pool_w,
                    self.stride,
                    self.padding)

        return dx

# LSTM


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
        
class TimeFC:
    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None

    def forward(self, x):
        W, b = self.params
        N, D = x.shape

        reshaped_x = x.reshape(N, -1)
        y = np.dot(reshaped_x, W) + b
        
        self.x = x
        y = y.reshape(N, -1)
        return y

    def backward(self, dy):
        W, b = self.params
        x = self.x
        N, D = x.shape

        dy = dy.reshape(N, -1)
        reshaped_x = x.reshape(N, -1)

        db = np.sum(dy, axis=0)
        dx = np.matmul(dy, W.T)
        dW = np.matmul(reshaped_x.T, dy)
        
        dx = dx.reshape(*x.shape)

        self.grads[0][...] = dW
        self.grads[1][...] = db

        return dx

class TimeMSE:
    def __init__(self):
        self.params, self.grads = [], []
        self.activation = ReLU()
        self.cache = None

    def forward(self, xs, ts):
        N, V = xs.shape
        xs = xs.reshape(N, V)
        xs = self.activation.forward(xs)
        ts = ts.reshape(N, V)

        loss = MSE(xs, ts)
        self.cache = (ts, xs, (N, V))

        return loss

    def backward(self, dy = 1):

        ts, xs, (N,  V) = self.cache
        
        dx = dy * (xs - ts) / (N)

        dx = self.activation.backward(dx)
        dx = dx.reshape(N , V)
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

        A = np.matmul(x, Wx) + np.matmul(h_prev, Wh) + b

        # slice for gates and get
        forget = A[:, :H]       # NxH
        get = A[:, H:2*H]
        input = A[:, 2*H:3*H]
        output = A[:, 3*H:4*H]

        forget = sigmoid(forget)   # forget gate
        get = np.tanh(get)        # new memory
        input = sigmoid(input)    # input gate
        output = sigmoid(output)    # output gate

        c_next = (c_prev * forget) + (get * input)
        h_next = np.tanh(c_next) * output

        self.cache = (x, h_prev, c_prev, input, forget, get, output, c_next)
        return h_next, c_next

    def backward(self, dh_next, dc_next):
        Wx, Wh, b = self.params
        x, h_prev, c_prev, input, forget, get, output, c_next = self.cache

        # chain rule
        do = dh_next * np.tanh(c_next)
        do_s = do * output*(1-output)
        dt = dh_next * output
        dt_c = dt * (1-(np.tanh(c_next)**2))

        di = dt_c * get
        dg = dt_c * input
        di_s = di * input*(1-input)
        dg_t = dg * (1-(get**2))

        dc_prev = dt_c * forget
        df = dt_c * c_prev
        df_s = df * forget*(1-forget)

        dA = np.hstack((df_s, dg_t, di_s, do_s))

        db = np.sum(dA, axis=0)
        dWh = np.matmul(h_prev.T, dA)
        dh_prev = np.matmul(dA, Wh.T)
        dWx = np.matmul(x.T, dA)
        dx = np.matmul(dA, Wx.T)

        self.grads[0][...] = dWx
        self.grads[1][...] = dWh
        self.grads[2][...] = db

        return dx, dh_prev, dc_prev


class TimeLSTM:
    def __init__(self, Wx, Wh, b, stateful=False):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.layers = None      # for LSTM layer
        self.h, self.c = None, None
        self.dh = None
        self.stateful = stateful

    def set_state(self, h, c=None):
        self.h, self.c = h, c

    def reset_state(self):
        self.h, self.c = None, None

    def forward(self, xs):
        Wx, Wh, b = self.params
        N, T, D = xs.shape      # mini-batch, time length, Dimension
        self.T = T
        H = Wh.shape[0]         # Wh (H, 4H) H: hidden size

        self.layers = []        # for stacking LSTM layer (horizontal)
        hs = np.empty((N, T, H), dtype='f')   # for save (h0 ... ht)

        # if not stateful, initialize h and c
        if not self.stateful or self.h is None:
            self.h = np.zeros((N, H), dtype='f')
        if not self.stateful or self.c is None:
            self.c = np.zeros((N, H), dtype='f')

        for t in range(T):
            layer = LSTM(*self.params)
            self.h, self.c = layer.forward(xs[:, t, :], self.h, self.c)

            hs[:, t, :] = self.h
            self.layers.append(layer)
        # many to one
        hs = hs[:, -1, :]  # return last states
        return hs

    def backward(self, dhs):
        Wx, Wh, b = self.params
        N, H = dhs.shape
        T = self.T
        D = Wx.shape[0]

        dxs = np.empty((N, T, D), dtype='f')
        dh, dc = 0, 0
        dh = dhs

        grads = [0, 0, 0]  # dWx, dWh, db
        for t in reversed(range(self.T)):  # BPTT
            layer = self.layers[t]
            # dx, dh, dc = layer.backward(dhs[:,t ,:] + dh, dc)
            dx, dh, dc = layer.backward(dh, dc)
            dxs[:, t, :] = dx
            for i, grad in enumerate(layer.grads):
                grads[i] += grad

        for i, grad in enumerate(grads):
            self.grads[i][...] = grad

        self.dh = dh
        return dxs

class GRU(object):
    def __init__(self, Wx, Wh, b):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.cache = None

    def affine(self, x, Wx, h, Wh, b):
        return np.matmul(x, Wx) + np.matmul(h, Wh) + b

    def forward(self, x, h_prev):
        # GRU doesn't have cell state
        Wx_, Wh_, b_ = self.params
        N, H = h_prev.shape

        Wxz, Wxr, Wx = Wx_[:, :H], Wx_[:, H:2*H], Wx_[:, 2*H:]
        Whz, Whr, Wh = Wh_[:, :H], Wh_[:, H:2*H], Wh_[:, 2*H:]
        bz, br, b = b_[:H], b_[H:2*H], b_[2*H:]

        z = sigmoid(self.affine(x, Wxz, h_prev, Whz, bz))
        r = sigmoid(self.affine(x, Wxr, h_prev, Whr, br))
        h_hat = np.tanh(self.affine(x, Wx, r*h_prev, Wh, b))
        h_next = (1-z)*h_prev + z*h_hat

        self.cache = (x, h_prev, z, r, h_hat)

        return h_next

    def backward(self, dh_next):
        x, h_prev, z, r, h_hat = self.cache
        Wx_, Wh_, b_ = self.params
        N, H = h_prev.shape

        Wxz, Wxr, Wx = Wx_[:, :H], Wx_[:, H:2*H], Wx_[:, 2*H:]
        Whz, Whr, Wh = Wh_[:, :H], Wh_[:, H:2*H], Wh_[:, 2*H:]

        dh_hat = z*dh_next
        dh_prev = (1-z)*dh_next

        # h_hat
        # y = h_hat
        # tanh diff = (1-y**2)
        dtanh = dh_hat*(1-h_hat**2)
        dx_hat = np.matmul(dtanh, Wx.T)
        dWx_hat = np.matmul(x.T, dtanh)
        dWh_hat = np.matmul((r*h_prev).T, dtanh)
        db_hat = np.sum(dtanh, axis=0)

        dh_r = np.matmul(dtanh, Wh.T)
        dh_prev_hat = r*dh_r

        # r
        dr = dh_r*h_prev
        drs = dr*r*(1-r)
        dx_r = np.matmul(drs, Wxr.T)
        dWx_r = np.matmul(x.T, drs)
        dh_r = np.matmul(drs, Whr.T)
        dWh_r = np.matmul(h_prev.T, drs)
        db_r = np.sum(drs, axis=0)

        # z
        dz = dh_next*h_hat - dh_next*h_prev
        dzs = dz*z*(1-z)
        dx_z = np.matmul(dzs, Wxz.T)
        dWx_z = np.matmul(x.T, dzs)
        dh_z = np.matmul(dzs, Whz.T)
        dWh_z = np.matmul(h_prev.T, dzs)
        db_z = np.sum(dzs, axis=0)

        dWx = np.hstack((dWx_hat, dWx_r, dWx_z))
        dWh = np.hstack((dWh_hat, dWh_r, dWh_z))
        db = np.hstack((db_hat, db_r, db_z))

        dx = dx_hat + dx_r + dx_z
        dh_prev += (dh_hat + dh_r + dh_z)

        self.grads[0][...] = dWx
        self.grads[1][...] = dWh
        self.grads[2][...] = db

        return dx, dh_prev


class TimeGRU(object):
    def __init__(self, Wx, Wh, b, stateful=False):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.layers = None
        self.h = None
        self.dh = None
        self.stateful = stateful

    def set_state(self, h):
        self.h = h

    def reset_state(self):
        self.h = None

    def forward(self, xs):
        # xs is x sequences
        Wx, Wh, b = self.params
        N, T, D = xs.shape  # batch, time step, dimension
        self.T = T
        H = Wh.shape[0]     # hidden size

        self.layers = []
        hs = np.empty((N, T, H), dtype='f')     # save h0 to ht

        # initialize h
        if not self.stateful or self.h is None:
            self.h = np.zeros((N, H), dtype='f')

        # loop for time step
        for t in range(T):
            layer = GRU(*self.params)
            self.h = layer.forward(xs[:, t, :], self.h)
            hs[:, t, :] = self.h
            self.layers.append(layer)
        
        # many to one
        hs = hs[:, -1, :]  # return last states
        
        return hs

    def backward(self, dhs):
        Wx, Wh, b = self.params
        N, H = dhs.shape
        T = self.T
        D = Wx.shape[0]

        dxs = np.empty((N, T, D), dtype='f')
        dh = 0
        dh = dhs

        grads = [0, 0, 0]       # grads for Wx, Wh, b
        for t in reversed(range(T)):    # backpropagation through time
            layer = self.layers[t]
            dx, dh = layer.backward(dh)
            dxs[:, t, :] = dx
            for i, grad in enumerate(layer.grads):
                grads[i] += grad

        for i, grad in enumerate(grads):
            self.grads[i][...] = grad

        self.dh = dh
        return dxs