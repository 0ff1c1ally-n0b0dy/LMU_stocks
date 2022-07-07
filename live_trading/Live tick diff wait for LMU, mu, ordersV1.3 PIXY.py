import urllib
import json
from urllib.request import urlopen
from urllib.error import HTTPError
from polygon import WebSocketClient, STOCKS_CLUSTER
from polygon import RESTClient
import time
import warnings
import concurrent.futures

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.order import *

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch
import torch.nn as nn
from torch import fft
from torch.nn import init
from torch.nn import functional as F
from datetime import datetime
import requests

warnings.filterwarnings('ignore')

key = 're02B0ew7bKEr8wZpFlhTFZp4Kjmh9j3'
ticker = "PIXY"

torch.cuda.is_available()
ngpu = 1
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")


################################33
# Create class for placing orders
#####################################

class TestApp(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)

    def error(self, reqId, errorCode, errorString):
        print("Error: ", reqId, " ", errorCode, " ", errorString)

    def nextValidId(self, orderId: int, ticker, action, quantity):
        super().nextValidId(orderId)

        # print("Setting nextValidId: %d"%(orderId))
        self.nextOrderId = orderId
        print("NextValidId:", orderId)
        self.start(ticker, action, quantity)

    def orderStatus(self, orderId, status, filled, remaining, lastFillPrice):
        print("OrderStatus. Id:", orderId, ", Status: ", status, ", Filled: ", filled, ", Remaining: ", remaining,
              ", Last Fill Price: ", lastFillPrice)

    def openOrder(self, orderId, contract, order):
        print("Open Order. ID: ", orderId, contract.symbol, contract.SecType, "@", contract.exchange, ":", order.action,
              order.orderType, order.TotalQuantity)

    def execDetails(self, reqId, contract, execution):
        print("ExecDetails. ", reqId, contract.symbol, contract.SecType, contract.currency, execution.execId,
              execution.orderId, execution.orderId, execution.shares, execution.LastLiquidity)

    def start(self, ticker: str, action: str, quantity: int):
        contract = Contract()
        contract.symbol = ticker
        contract.SecType = "STK"
        contract.exchange = "SMART"
        contract.currency = "USD"
        contract.primaryExchange = "NASDAQ"

        order = Order()
        order.action = action
        order.orderType = "MKT"
        order.totalQuantity = quantity

        self.placeOrder(self.nextOrderId, contract, order)

    def stop(self):
        self.done = True
        self.disconnect()


app = TestApp()
client_id = 12345
app.connect("127.0.0.1", 7497, client_id)
app.nextOrderId = 0


###########################
# Create LMU
#########################

def leCunUniform(tensor):
    """
        LeCun Uniform Initializer
        References:
        [1] https://keras.rstudio.com/reference/initializer_lecun_uniform.html
        [2] Source code of _calculate_correct_fan can be found in https://pytorch.org/docs/stable/_modules/torch/nn/init.html
        [3] Yann A LeCun, Léon Bottou, Genevieve B Orr, and Klaus-Robert Müller. Efficient backprop. In Neural networks: Tricks of the trade, pages 9–48. Springer, 2012
    """
    fan_in = init._calculate_correct_fan(tensor, "fan_in")
    limit = np.sqrt(3. / fan_in)
    init.uniform_(tensor, -limit, limit)  # fills the tensor with values sampled from U(-limit, limit)


class LMUCell(nn.Module):
    """ A single LMU Cell """

    def __init__(self, input_size, hidden_size, memory_size, theta, learn_a=False, learn_b=False, psmnist=False):
        """
        Parameters:
            input_size (int) :
                Size of the input vector (x_t)
            hidden_size (int) :
                Size of the hidden vector (h_t)
            memory_size (int) :
                Size of the memory vector (m_t)
            theta (int) :
                The number of timesteps in the sliding window that is represented using the LTI system
            learn_a (boolean) :
                Whether to learn the matrix A (default = False)
            learn_b (boolean) :
                Whether to learn the matrix B (default = False)
            psmnist (boolean) :
                Uses different parameter initializers when training on psMNIST (as specified in the paper)
        """

        super(LMUCell, self).__init__()

        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.f = nn.Tanh()

        A, B = self.stateSpaceMatrices(memory_size, theta)
        A = torch.from_numpy(A).float()
        B = torch.from_numpy(B).float()

        if learn_a:
            self.A = nn.Parameter(A)
        else:
            self.register_buffer("A", A)

        if learn_b:
            self.B = nn.Parameter(B)
        else:
            self.register_buffer("B", B)

        # Declare Model parameters:
        ## Encoding vectors
        self.e_x = nn.Parameter(torch.empty(1, input_size))
        self.e_h = nn.Parameter(torch.empty(1, hidden_size))
        self.e_m = nn.Parameter(torch.empty(1, memory_size))
        ## Kernels
        self.W_x = nn.Parameter(torch.empty(hidden_size, input_size))
        self.W_h = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.W_m = nn.Parameter(torch.empty(hidden_size, memory_size))

        self.initParameters(psmnist)

    def initParameters(self, psmnist=False):
        """ Initialize the cell's parameters """

        if psmnist:
            # Initialize encoders
            leCunUniform(self.e_x)
            init.constant_(self.e_h, 0)
            init.constant_(self.e_m, 0)
            # Initialize kernels
            init.constant_(self.W_x, 0)
            init.constant_(self.W_h, 0)
            init.xavier_normal_(self.W_m)
        else:
            # Initialize encoders
            leCunUniform(self.e_x)
            leCunUniform(self.e_h)
            init.constant_(self.e_m, 0)
            # Initialize kernels
            init.xavier_normal_(self.W_x)
            init.xavier_normal_(self.W_h)
            init.xavier_normal_(self.W_m)

    def stateSpaceMatrices(self, memory_size, theta):
        """ Returns the discretized state space matrices A and B """

        Q = np.arange(memory_size, dtype=np.float64).reshape(-1, 1)
        R = (2 * Q + 1) / theta
        i, j = np.meshgrid(Q, Q, indexing="ij")

        # Continuous
        A = R * np.where(i < j, -1, (-1.0) ** (i - j + 1))
        B = R * ((-1.0) ** Q)
        C = np.ones((1, memory_size))
        D = np.zeros((1,))

        # Convert to discrete
        A, B, C, D, dt = cont2discrete(
            system=(A, B, C, D),
            dt=1.0,
            method="zoh"
        )

        return A, B

    def forward(self, x, state):
        """
        Parameters:
            x (torch.tensor):
                Input of size [batch_size, input_size]
            state (tuple):
                h (torch.tensor) : [batch_size, hidden_size]
                m (torch.tensor) : [batch_size, memory_size]
        """

        h, m = state

        # Equation (7) of the paper
        u = F.linear(x, self.e_x) + F.linear(h, self.e_h) + F.linear(m, self.e_m)  # [batch_size, 1]

        # Equation (4) of the paper
        m = F.linear(m, self.A) + F.linear(u, self.B)  # [batch_size, memory_size]

        # Equation (6) of the paper
        h = self.f(
            F.linear(x, self.W_x) +
            F.linear(h, self.W_h) +
            F.linear(m, self.W_m)
        )  # [batch_size, hidden_size]

        return h, m


class LMU(nn.Module):
    """ An LMU layer """

    def __init__(self, input_size, hidden_size, memory_size, theta, learn_a=False, learn_b=False, psmnist=False):
        """
        Parameters:
            input_size (int) :
                Size of the input vector (x_t)
            hidden_size (int) :
                Size of the hidden vector (h_t)
            memory_size (int) :
                Size of the memory vector (m_t)
            theta (int) :
                The number of timesteps in the sliding window that is represented using the LTI system
            learn_a (boolean) :
                Whether to learn the matrix A (default = False)
            learn_b (boolean) :
                Whether to learn the matrix B (default = False)
            psmnist (boolean) :
                Uses different parameter initializers when training on psMNIST (as specified in the paper)
        """

        super(LMU, self).__init__()
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.cell = LMUCell(input_size, hidden_size, memory_size, theta, learn_a, learn_b, psmnist)

    def forward(self, x, state=None):
        """
        Parameters:
            x (torch.tensor):
                Input of size [batch_size, seq_len, input_size]
            state (tuple) : (default = None)
                h (torch.tensor) : [batch_size, hidden_size]
                m (torch.tensor) : [batch_size, memory_size]
        """

        # Assuming batch dimension is always first, followed by seq. length as the second dimension
        batch_size = x.size(0)
        seq_len = x.size(1)

        # Initial state (h_0, m_0)
        if state == None:
            h_0 = torch.zeros(batch_size, self.hidden_size)
            m_0 = torch.zeros(batch_size, self.memory_size)
            if x.is_cuda:
                h_0 = h_0.cuda()
                m_0 = m_0.cuda()
            state = (h_0, m_0)

        # Iterate over the timesteps
        output = []
        for t in range(seq_len):
            x_t = x[:, t, :]  # [batch_size, input_size]
            h_t, m_t = self.cell(x_t, state)
            state = (h_t, m_t)
            output.append(h_t)

        output = torch.stack(output)  # [seq_len, batch_size, hidden_size]
        output = output.permute(1, 0, 2)  # [batch_size, seq_len, hidden_size]

        return output, state  # state is (h_n, m_n) where n = seq_len


class LMUFFT(nn.Module):
    def __init__(self, input_size, hidden_size, memory_size, seq_len, theta):
        super(LMUFFT, self).__init__()

        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.seq_len = seq_len
        self.theta = theta

        self.W_u = nn.Linear(in_features=input_size, out_features=1)
        self.f_u = nn.ReLU()
        self.W_h = nn.Linear(in_features=memory_size + input_size, out_features=hidden_size)
        self.f_h = nn.ReLU()

        A, B = self.stateSpaceMatrices()
        self.register_buffer("A", A)  # [memory_size, memory_size]
        self.register_buffer("B", B)  # [memory_size, 1]

        H, fft_H = self.impulse()
        self.register_buffer("H", H)  # [memory_size, seq_len]
        self.register_buffer("fft_H", fft_H)  # [memory_size, seq_len + 1]

    def stateSpaceMatrices(self):
        """ Returns the discretized state space matrices A and B """

        Q = np.arange(self.memory_size, dtype=np.float64).reshape(-1, 1)
        R = (2 * Q + 1) / self.theta
        i, j = np.meshgrid(Q, Q, indexing="ij")

        # Continuous
        A = R * np.where(i < j, -1, (-1.0) ** (i - j + 1))
        B = R * ((-1.0) ** Q)
        C = np.ones((1, self.memory_size))
        D = np.zeros((1,))

        # Convert to discrete
        A, B, C, D, dt = cont2discrete(
            system=(A, B, C, D),
            dt=1.0,
            method="zoh"
        )

        # To torch.tensor
        A = torch.from_numpy(A).float()  # [memory_size, memory_size]
        B = torch.from_numpy(B).float()  # [memory_size, 1]

        return A, B

    def impulse(self):
        """ Returns the matrices H and the 1D Fourier transform of H (Equations 23, 26 of the paper) """

        H = []
        A_i = torch.eye(self.memory_size)
        for t in range(self.seq_len):
            H.append(A_i @ self.B)
            A_i = self.A @ A_i

        H = torch.cat(H, dim=-1)  # [memory_size, seq_len]
        fft_H = fft.rfft(H, n=2 * self.seq_len, dim=-1)  # [memory_size, seq_len + 1]

        return H, fft_H

    def forward(self, x):
        """
        Parameters:
            x (torch.tensor):
                Input of size [batch_size, seq_len, input_size]
        """

        batch_size, seq_len, input_size = x.shape

        # Equation 18 of the paper
        u = self.f_u(self.W_u(x))  # [batch_size, seq_len, 1]

        # Equation 26 of the paper
        fft_input = u.permute(0, 2, 1)  # [batch_size, 1, seq_len]
        fft_u = fft.rfft(fft_input, n=2 * seq_len, dim=-1)  # [batch_size, seq_len, seq_len+1]

        # Element-wise multiplication (uses broadcasting)
        # [batch_size, 1, seq_len+1] * [1, memory_size, seq_len+1]
        temp = fft_u * self.fft_H.unsqueeze(0)  # [batch_size, memory_size, seq_len+1]

        m = fft.irfft(temp, n=2 * seq_len, dim=-1)  # [batch_size, memory_size, seq_len+1]
        m = m[:, :, :seq_len]  # [batch_size, memory_size, seq_len]
        m = m.permute(0, 2, 1)  # [batch_size, seq_len, memory_size]

        # Equation 20 of the paper (W_m@m + W_x@x  W@[m;x])
        input_h = torch.cat((m, x), dim=-1)  # [batch_size, seq_len, memory_size + input_size]
        h = self.f_h(self.W_h(input_h))  # [batch_size, seq_len, hidden_size]

        h_n = h[:, -1, :]  # [batch_size, hidden_size]

        return h, h_n


class Model(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, memory_size, seq_len, theta):
        super().__init__()
        self.lmu_fft = LMUFFT(input_size, hidden_size, memory_size, seq_len, theta)
        self.linear = nn.Linear(hidden_size, output_size)
        # self.dropout=nn.Dropout(p=0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _, h_n = self.lmu_fft(x)  # [batch_size, hidden_size]
        # h_n = self.dropout(h_n)
        # print(x.size())
        output = self.linear(h_n)
        output = self.sigmoid(output)
        output = output.view(1)
        return output  # [batch_size, output_size]


######################
# Load the saved weights in the next line
######################
model = torch.load(
    "LMU trade&tick diff, time in epochs 5, accuracy 88.8407%,hidden 32,memory 32,theta 512,batch 1,window 2000.pth")

seq_len = window = 2000
input_size = n_var = 8
hidden_size = 32
memory_size = 32
theta = 512
output_size = 1
epochs = 1
lr = 10 ** (-4)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.BCELoss()

# needed to run in parallel on multiple GPUs.
if (device.type == "cuda") and (ngpu > 1):
    model = nn.DataParallel(model, list(range(ngpu)))
    model = model.to(device)


def timing(since):
    now = time.time()
    s = now - since
    m = int(s / 60)
    s = s - m * 60
    return "%dm %ds" % (m, s)


def max_ask(s, gamma, q, sigma, mu, capital):
    q_max = int(capital / s)
    omega = 0.5 * (gamma ** 2) * (sigma ** 2) * (1 + q_max) ** 2 + gamma * (1 + q_max) * mu
    return s + (1 / gamma) * np.log(1 + ((1 - 2 * q) * (gamma ** 2) * (sigma ** 2) - gamma * mu) / (
                2 * omega - (gamma ** 2) * (q ** 2) * sigma ** 2 + 2 * gamma * q * mu))


def min_bid(s, gamma, q, sigma, mu, capital):
    q_max = int(capital / s)
    omega = 0.5 * (gamma ** 2) * (sigma ** 2) * (1 + q_max) ** 2 + gamma * (1 + q_max) * mu
    return s + (1 / gamma) * np.log(1 + (2 * gamma * mu - (1 + 2 * q) * (gamma ** 2) * (sigma ** 2)) / (
                2 * omega + 2 * gamma * q * mu - (gamma ** 2) * (q ** 2) * sigma ** 2))


def combine_data(idx, last_tick_index=0):
    seconds_trades_idx = seconds_trades[idx]
    for j in range(last_tick_index, nr_ticks - 1):
        if seconds_trades_idx < seconds_ticks[0]:
            return ask[0], ask_size[0], bid[0], bid_size[0]
        if seconds_ticks[j] <= seconds_trades_idx and seconds_ticks[j + 1] > seconds_trades_idx:
            last_tick_index = j
            return ask[j], ask_size[j], bid[j], bid_size[j]
        if seconds_trades_idx >= seconds_ticks[nr_ticks - 1]:
            return ask[nr_ticks - 1], ask_size[nr_ticks - 1], bid[nr_ticks - 1], bid_size[nr_ticks - 1]


date = datetime.today().strftime("%Y-%m-%d")
# date="2021-12-27"
buffer = 10
limit_ticks = window + buffer
p_thr = 0.5

#######################
# Initially, I need to extract at least window+10 (confirmation size) trades from polygon and run the LMU on those.
# Memorize the time of the last trade I extracted and after the LMU is done for this first step, download trade data from the
# last time until present, run the LMU on this data, and repeat.
######################

start = time.time()
flag = 1
flag_404 = 1
flag_400 = 1
count_502 = 0
link_ticks = "https://api.polygon.io/v2/ticks/stocks/nbbo/%s/%s?reverse=true&limit=%d&apiKey=%s" % (
ticker, date, limit_ticks, key)
while flag == 1 and count_502 <= 10:
    try:
        response_ticks = urllib.request.urlopen(link_ticks)
        flag = 0
    except urllib.error.HTTPError as err:
        if err.code == 502:
            time.sleep(1)
            flag = 1
            count_502 += 1
        else:
            if err.code == 404:
                flag_404 = 0
                flag = 0
            else:
                if err.code == 400:
                    flag_400 = 0
                    flag = 0
                else:
                    flag = 0
if flag == 0 and flag_404 == 1 and flag_400 == 1:
    response_ticks = urlopen(link_ticks)
    data_json_ticks = json.loads(response_ticks.read())
    nr_ticks = len(data_json_ticks["results"])

data = pd.DataFrame()

seconds_ticks = [None] * nr_ticks
ask = [None] * nr_ticks
ask_size = [None] * nr_ticks
bid = [None] * nr_ticks
bid_size = [None] * nr_ticks

for i in range(nr_ticks):
    seconds_ticks[i] = data_json_ticks["results"][nr_ticks - 1 - i]["t"]
    ask[i] = data_json_ticks["results"][nr_ticks - 1 - i]["P"]
    ask_size[i] = data_json_ticks["results"][nr_ticks - 1 - i]["S"]
    bid[i] = data_json_ticks["results"][nr_ticks - 1 - i]["p"]
    bid_size[i] = data_json_ticks["results"][nr_ticks - 1 - i]["s"]

last_time = seconds_ticks[-1]

data.dropna(axis=0)

seconds_diff = [seconds_ticks[idx + 1] - seconds_ticks[idx] for idx in range(nr_ticks - 1)]
data = pd.DataFrame(data, index=range(1, nr_ticks))

data["seconds diff"] = seconds_diff
data["ask." + ticker] = ask
data["ask_size." + ticker] = ask_size
data["bid." + ticker] = bid
data["bid_size." + ticker] = bid_size

print(data.tail())

scaled_data = data.copy()
columns = list(scaled_data.columns)
n_cols = len(columns)

for k in range(n_cols):
    scaler = MinMaxScaler()
    scaled_data[columns[k]] = scaler.fit_transform(np.array(scaled_data[columns[k]]).reshape(-1, 1))

n_test = len(scaled_data)
scaled_data["events." + ticker] = [0] * n_test
data["events." + ticker] = [0] * n_test
batch_size = 1
outputs = []
gamma = 10 ** (-1)

for j in range(n_test - window):
    x_test = scaled_data.values[j:(j + window), :]
    x_test = torch.FloatTensor(x_test).to(device)
    if x_test.size(0) != 0:
        x_test = x_test.view(batch_size, window, n_var)
        y_pred = model(x_test)

        if y_pred >= p_thr:
            output_class = 1
        else:
            output_class = 0
        outputs.append(output_class)

        scaled_data["events." + ticker].iloc[j + window] = output_class
        data["events." + ticker].iloc[j + window] = output_class

nr_ones_buy=10
count_iter = 0
total_iter = 10
total_nr_shares = 0
profit_start = 0
profit_end = 0
profits = []
returns = []
time_start = []
time_end = []
entry_prices = []
row_labels = []
capital = 10 ** 7
remaining_capital = capital
nr_shares = 0
nr_buy_trades = 0
nr_sell_trades = 0
print_csv_iter = 100
profit_data = pd.DataFrame()

while True:
    start = time.time()
    flag = 1
    flag_404 = 1
    flag_400 = 1
    count_502 = 0
    link_ticks = "https://api.polygon.io/v2/ticks/stocks/nbbo/%s/%s?timestamp=%d&reverse=false&apiKey=%s" % (
    ticker, date, last_time, key)
    while flag == 1 and count_502 <= 10:
        try:
            response_ticks = urllib.request.urlopen(link_ticks)
            flag = 0
        except urllib.error.HTTPError as err:
            if err.code == 502:
                time.sleep(1)
                flag = 1
                count_502 += 1
            else:
                if err.code == 404:
                    flag_404 = 0
                    flag = 0
                else:
                    if err.code == 400:
                        flag_400 = 0
                        flag = 0
                    else:
                        flag = 0
    if flag == 0 and flag_404 == 1 and flag_400 == 1:
        response_ticks = urlopen(link_ticks)
        data_json_ticks = json.loads(response_ticks.read())
        nr_ticks = len(data_json_ticks["results"])

    if nr_ticks > 1:
        data = data.iloc[(buffer - 1):]
        n_rows = len(data)
        seconds_ticks = [None] * nr_ticks
        ask = [None] * nr_ticks
        ask_size = [None] * nr_ticks
        bid = [None] * nr_ticks
        bid_size = [None] * nr_ticks

        for i in range(nr_ticks):
            seconds_ticks[i] = data_json_ticks["results"][nr_ticks - 1 - i]["t"]
            ask[i] = data_json_ticks["results"][nr_ticks - 1 - i]["P"]
            ask_size[i] = data_json_ticks["results"][nr_ticks - 1 - i]["S"]
            bid[i] = data_json_ticks["results"][nr_ticks - 1 - i]["p"]
            bid_size[i] = data_json_ticks["results"][nr_ticks - 1 - i]["s"]

        print("[%d] Nr ticks happened since last download, combination and LMU feeding: %d" % (count_iter, nr_ticks))

        seconds_ticks.insert(0, last_time)
        seconds_diff = [seconds_ticks[idx + 1] - seconds_ticks[idx] for idx in range(nr_ticks)]

        data_xtra = pd.DataFrame()
        data_xtra["seconds diff"] = seconds_diff

        last_time = seconds_ticks[-1]

        data_xtra["ask." + ticker] = ask
        data_xtra["ask_size." + ticker] = ask_size
        data_xtra["bid." + ticker] = bid
        data_xtra["bid_size." + ticker] = bid_size

        data = data.append(data_xtra, ignore_index=True)
        # data.dropna(axis=0)
        print(data.tail())

        n_rows = len(data)
        data = pd.DataFrame(data, index=range(1, n_rows))

        scaled_data = data.copy()
        columns = list(scaled_data.columns)
        n_cols = len(columns)

        for k in range(n_cols):
            if columns[k] != "events." + ticker:
                scaler = MinMaxScaler()
                scaled_data[columns[k]] = scaler.fit_transform(np.array(scaled_data[columns[k]]).reshape(-1, 1))

        n_test = len(scaled_data)
        batch_size = 1

        outputs = []
        for j in range(n_test - window):
            x_test = scaled_data.values[j:(j + window), :]
            x_test = torch.FloatTensor(x_test).to(device)
            if x_test.size(0) != 0:
                x_test = x_test.view(batch_size, window, n_var)
                y_pred = model(x_test)

                if y_pred >= p_thr:
                    output_class = 1
                else:
                    output_class = 0
                # all_outputs.append(output_class)
                outputs.append(output_class)

                scaled_data["events." + ticker].iloc[j + window] = output_class
                data["events." + ticker].iloc[j + window] = output_class

        buffer = nr_ticks
        count_iter += 1
        print("[%d] Time for iteration is %s" % (count_iter, timing(start)))
        print("Outputs are: ")
        print(outputs)

    else:
        buffer = 1
        time.sleep(1)

    #########################333
    # Place order according to paper idea first
    ###############################

    # Problem 1: sigma and mu are computed over the small snapshot between LMU predictions, whereas in forward testing it was over the whole history
    # Can't do it over the whole history here because it would slow down the code. I will try it this way.

    len_outputs = len(outputs)
    len_data=len(data["events." + ticker])
    sigma = np.std((data["ask." + ticker] + data["bid." + ticker]) / 2)
    mu = np.mean((data["ask." + ticker] + data["bid." + ticker]) / 2)
    s = (data["ask." + ticker].iloc[-1] + data["bid." + ticker].iloc[-1]) / 2
    max_ask_price = max_ask(s=s, gamma=gamma, q=total_nr_shares, mu=mu, sigma=sigma, capital=capital)
    min_bid_price = min_bid(s=s, gamma=gamma, q=total_nr_shares, mu=mu, sigma=sigma, capital=capital)
    indiff_price = (max_ask_price + min_bid_price) / 2

    if sum(data["events." + ticker].iloc[(len_data - nr_ones_buy - 1):(len_data - 1)]) != None:
        if sum(data["events." + ticker].iloc[
               (len_data - nr_ones_buy - 1):(len_data - 1)]) == nr_ones_buy and s <= indiff_price and (
                data["ask." + ticker].iloc[-1] > 0 or data["bid." + ticker].iloc[-1] > 0):
            print("[%s] Entered position" % (ticker))
            if data["ask." + ticker].iloc[-1] * data["ask_size." + ticker].iloc[-1] > capital:
                nr_shares = int(capital / data["ask." + ticker].iloc[-1])
                profit_start += data["ask." + ticker].iloc[-1] * nr_shares

                # IB place buy order
                app.nextOrderId+=1
                app.start(ticker, action="BUY", quantity=nr_shares)
                #app.openOrder(app.nextOrderId,contract,order)

                nr_buy_trades += 1
                entry_prices.append(profit_start)
                time_start.append(seconds_ticks[-1])
                remaining_capital -= data["ask." + ticker].iloc[-1] * nr_shares
            else:
                nr_shares = data["ask_size." + ticker].iloc[-1]
                profit_start += data["ask." + ticker].iloc[-1] * nr_shares

                # IB place buy order
                app.nextOrderId+=1
                app.start(ticker, action="BUY", quantity=nr_shares)
                #app.openOrder(app.nextOrderId,contract,order)

                nr_buy_trades += 1
                entry_prices.append(profit_start)
                time_start.append(seconds_ticks[-1])
                remaining_capital -= data["ask." + ticker].iloc[-1] * nr_shares

            total_nr_shares += nr_shares

        if total_nr_shares > 0 and sum(
                data["events." + ticker].iloc[(len_data - nr_ones_buy - 1):(len_data - 1)]) == 0 and (
                data["ask." + ticker].iloc[-1] > 0 or data["bid." + ticker].iloc[-1] > 0):
            while total_nr_shares > 0:
                if data["bid_size." + ticker].iloc[-1] < total_nr_shares:
                    nr_shares = data["bid_size." + ticker].iloc[-1]
                    profit_end += data["bid." + ticker].iloc[-1] * nr_shares
                    total_nr_shares -= nr_shares

                    # IB place sell order
                    app.nextOrderId+=1
                    app.start(ticker, action="SELL", quantity=nr_shares)
                    #app.openOrder(app.nextOrderId,contract,order)

                    nr_sell_trades += 1
                    remaining_capital += data["bid." + ticker].iloc[-1] * nr_shares
                else:
                    profit_end += data["bid." + ticker].iloc[-1] * total_nr_shares
                    time_end.append(seconds_ticks[-1])

                    # IB place sell order
                    app.nextOrderId+=1
                    app.start(ticker, action="SELL", quantity=total_nr_shares)
                    #app.openOrder(app.nextOrderId,contract,order)

                    nr_sell_trades += 1
                    remaining_capital += data["bid." + ticker].iloc[-1] * total_nr_shares
                    total_nr_shares = 0

            if total_nr_shares == 0:
                profits = (profit_end - profit_start)
                remaining_capital += profits
                returns = (profit_end - profit_start) / profit_start

                profit_start = 0
                profit_end = 0

                row_labels.append("%s.trade %d" % (ticker, nr_buy_trades + nr_sell_trades))
                print("Return=%f, profit=%f, nr buy trades=%d, nr sell trades=%d" % (
                    returns, profits, nr_buy_trades, nr_sell_trades))
    else:
        buffer = 1
        time.sleep(1)
    if count_iter % print_csv_iter == 0 and len(row_labels) > 0:
        profit_data["stock.trade"] = row_labels
        profit_data["profit"] = profits
        profit_data["return"] = returns
        profit_data["time in trade"] = time_end - time_start
        profit_data.to_csv("Profits after iteration %d" % (count_iter))




