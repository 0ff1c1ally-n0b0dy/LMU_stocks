import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch
import torch.nn as nn
from torch import fft
from torch.nn import init
from torch.nn import functional as F
import os
import glob
import random
import warnings
from Utils import NamedQueue, timming, max_ask, min_bid
import datetime


import time
import matplotlib.pyplot as plt
from functools import partial

warnings.filterwarnings('ignore')

from scipy.signal import cont2discrete

#torch.cuda.is_available()
#ngpu = 1
#device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

device=torch.device("cpu")
###############3
#Forward testing folder
############
fwd_folder="D1 DataID007 forward test"    #this is what I saw it was called in your forward testing code

def create_seq(input_data, tw):
    x = []
    y = []
    Y = pd.DataFrame(input_data, columns=["events." + ticker])
    # X=input_data
    n_rows_input = input_data.shape[0]
    for i in range(n_rows_input - tw):
        seq = input_data.values[i:i + tw, :]
        target = Y.values[i + tw]
        x.append(seq)
        y.append(target)

    return x, y

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
        self.dropout = nn.Dropout(p=0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _, h_n = self.lmu_fft(x)  # [batch_size, hidden_size]
        h_n = self.dropout(h_n)
        output = self.linear(h_n)
        output = self.sigmoid(output)
        output = output.view(1)
        return output  # [batch_size, output_size]


seq_len = window = 1000
input_size = n_var = 3
hidden_size = 32
memory_size = 32
theta = 512
output_size = 1
model = Model(input_size=input_size, output_size=output_size, hidden_size=hidden_size,
              memory_size=memory_size, seq_len=seq_len, theta=theta).to(device)

criterion = nn.BCELoss()

# needed to run in parallel on multiple GPUs.
if (device.type == "cuda") and (ngpu > 1):
    model = nn.DataParallel(model, list(range(ngpu)))
    model = model.to(device)

epochs = 2
# printing_iter=10
p_thr = 0.5
percentage_test = 0.2
total_iters = 0
overall_accuracy = 0
batch_size = 1
count_training = 0
count_testing = 0
accuracies = []

###################
# Locations for the downloaded trade data. All folders have to be at the same level as this code.
###################

file1_text = "./Prints/Forward test no events, fixed lr: epochs %d, hidden %d,memory %d,theta %d,window %d.txt" % (
epochs, hidden_size, memory_size, theta, window)
file1 = open(file1_text, "w")
flag_error = 1

###########################
# Forward testing starts using the optimized weights from the Saved_weights folder in this case
# situated at the same level as this code.
###########################
model=torch.load("./Saved_weights/Batch2 LMU trade no events, epochs 3, accuracy 49.8188%,hidden 32,memory 32,theta 512,batch 1,window 1000.pth",
                 map_location='cpu')

################3
# Load the folder with csvs for forward testing (whole stock or per event).
###################
csvs = glob.glob("./" + fwd_folder + "/*.csv")
nr_testing_csv = len(csvs)
test_loss = 0
overall_accuracy = 0
test_losses = []
printing_iters = 10 ** 2
count_completion = 0

shares = 1
total_profits = []
profits = []
total_returns = []
returns = []
total_times = []
times = []
nr_trades = []
total_trades = []
row_labels = []
tickers = []
entry_prices = []
exit_prices = []
accuracies = []
time_start = []
time_end = []
file1.write("Forward testing starts for %s \n" % (fwd_folder))
print("Forward testing starts for %s" % (fwd_folder))
for file_name in csvs:
    start = time.time()
    ticker = file_name.split("/")[2].split(" ")[0]
    tickers.append(ticker)
    flag_error = 1
    count_completion += 1
    try:
        data = pd.read_csv(file_name, low_memory=True,
                           usecols=["seconds", "size." + ticker, "price." + ticker, "events." + ticker],
                           dtype={"size." + ticker: "float32", "price." + ticker: "float32",
                                  "events." + ticker: "int8"})
    except ValueError:
        flag_error = 0
        print("For %s the columns do not have the ticker's name" % (file_name.split("/")[2]))
    length_data = len(data)
    if flag_error == 1:
        if length_data > window:
            targets = []
            outputs = []
            list_outputs = []
            accuracy = 0
            test_iters = 0
            scaled_data = data.copy()
            scaled_data = scaled_data[["seconds", "size." + ticker, "price." + ticker]]
            columns = list(scaled_data.columns)
            n_var = len(columns)
            for k in range(n_var):
                scaler = MinMaxScaler()
                scaled_data[columns[k]] = scaler.fit_transform(np.array(scaled_data[columns[k]]).reshape(-1, 1))

            n_test = len(scaled_data)
            for j in range(n_test - window):
                x_test = scaled_data.values[j:(j + window), :]
                x_test = torch.FloatTensor(x_test).to(device)
                if x_test.size(0) != 0:
                    test_iters += 1
                    x_test = x_test.view(batch_size, window, n_var)

                    y_test = data["events." + ticker].iloc[j + window]
                    y_test = torch.FloatTensor([y_test]).to(device)

                    y_pred = model(x_test)
                    targets.append(int(y_test))

                    loss = criterion(y_pred, y_test)
                    test_loss += loss.item()

                    if y_pred >= p_thr:
                        output_class = 1
                    else:
                        output_class = 0
                    outputs.append(output_class)

                    if output_class == int(y_test):
                        accuracy += 1
            accuracy = accuracy / (n_test - window)
            accuracies.append(accuracy)

            test_losses.append(test_loss / (n_test - window))
            overall_accuracy += accuracy

            list_targets = targets
            list_outputs = [0] * (len(outputs) + 2)
            list_outputs[1:(len(outputs) + 1)] = outputs

            len_test = len(list_outputs)

            list_outputs_diff = [list_outputs[i] - list_outputs[i + 1] for i in range(len_test - 1)]
            entry_index = [i + window for i in range(len_test - 1) if list_outputs_diff[i] == -1]
            exit_index = [i + window - 1 for i in range(len_test - 1) if list_outputs_diff[i] == 1]

            capital = 10 ** (6)
            remaining_capital = capital

            count_trades = 0
            len_entries_index = len(entry_index)
            gamma = 10 ** (-1)
            if len_entries_index > 0:
                i = 0
                while i <= len_entries_index - 1:
                    if entry_index[i] < window:
                        i += 1
                    else:
                        nr_shares = 0
                        profit_start = 0
                        sigma = np.std(data["price." + ticker].iloc[(entry_index[i] - window):entry_index[i]])
                        mu = np.mean(data["price." + ticker].iloc[(entry_index[i] - window):entry_index[i]])
                        s = data["price." + ticker].iloc[entry_index[i]]
                        max_ask_price = max_ask(s=s, gamma=gamma, q=nr_shares, mu=mu, sigma=sigma,
                                                capital=remaining_capital)
                        min_bid_price = min_bid(s=s, gamma=gamma, q=nr_shares, mu=mu, sigma=sigma,
                                                capital=remaining_capital)
                        indiff_price = (max_ask_price + min_bid_price) / 2
                        file1.write("Entry index %d, exit index %d, price: %f and cutoff:%f \n" %
                                    (entry_index[i], exit_index[i], data["price." + ticker].iloc[entry_index[i]],
                                     indiff_price))
                        print("Entry index %d, exit index %d, price: %f and cutoff:%f" %
                              (entry_index[i], exit_index[i], data["price." + ticker].iloc[entry_index[i]],
                               indiff_price))
                        if exit_index[i] - entry_index[i] >= 10 and data["price." + ticker].iloc[
                            entry_index[i]] > 0 and \
                                data["price." + ticker].iloc[entry_index[i]] <= indiff_price:
                            file1.write("[%s] Entered position \n" % (ticker))
                            print("[%s] Entered position" % (ticker))
                            remaining_capital = capital
                            if data["price." + ticker].iloc[entry_index[i]] * data["size." + ticker].iloc[
                                entry_index[i]] > capital:
                                profit_start = data["price." + ticker].iloc[entry_index[i]] * int(
                                    capital / data["price." + ticker].iloc[entry_index[i]])
                                nr_shares += int(capital / data["price." + ticker].iloc[entry_index[i]])
                                file1.write("Baught %d shares at %f price<=%f, total nr shares is %d \n" %
                                            (data["size." + ticker].iloc[entry_index[i]],
                                             data["price." + ticker].iloc[entry_index[i]],
                                             indiff_price, nr_shares))
                                print("Baught %d shares at %f price<=%f, total nr shares is %d" %
                                      (data["size." + ticker].iloc[entry_index[i]],
                                       data["price." + ticker].iloc[entry_index[i]],
                                       indiff_price, nr_shares))
                                entry_prices.append(profit_start)
                                time_start.append(data["seconds"].iloc[entry_index[i] + 1])
                                remaining_capital -= data["price." + ticker].iloc[entry_index[i]] * int(
                                    capital / data["price." + ticker].iloc[entry_index[i]])
                            else:
                                profit_start = data["price." + ticker].iloc[entry_index[i]] * \
                                               data["size." + ticker].iloc[
                                                   entry_index[i]]
                                nr_shares += data["size." + ticker].iloc[entry_index[i]]
                                file1.write("Baught %d shares at %f price <= %f, total nr shares is %d \n" %
                                            (data["size." + ticker].iloc[entry_index[i]],
                                             data["price." + ticker].iloc[entry_index[i]],
                                             indiff_price, nr_shares))
                                print("Baught %d shares at %f price <= %f, total nr shares is %d" %
                                      (data["size." + ticker].iloc[entry_index[i]],
                                       data["price." + ticker].iloc[entry_index[i]],
                                       indiff_price, nr_shares))
                                entry_prices.append(profit_start)
                                time_start.append(data["seconds"].iloc[entry_index[i] + 1])
                                remaining_capital -= data["price." + ticker].iloc[entry_index[i]] * \
                                                     data["size." + ticker].iloc[entry_index[i]]
                                j = entry_index[i] + 1
                                while remaining_capital > 0 and j <= exit_index[i] and data["price." + ticker].iloc[
                                    j] <= indiff_price:
                                    sigma = np.std(data["price." + ticker].iloc[(j - window):j])
                                    mu = np.mean(data["price." + ticker].iloc[(j - window):j])
                                    max_ask_price = max_ask(s=s, gamma=gamma, q=nr_shares, mu=mu, sigma=sigma,
                                                            capital=remaining_capital)
                                    min_bid_price = min_bid(s=s, gamma=gamma, q=nr_shares, mu=mu, sigma=sigma,
                                                            capital=remaining_capital)
                                    indiff_price = (max_ask_price + min_bid_price) / 2
                                    if data["price." + ticker].iloc[j] * data["size." + ticker].iloc[
                                        j] > remaining_capital:
                                        profit_start += data["price." + ticker].iloc[j] * int(
                                            remaining_capital / data["price." + ticker].iloc[j])
                                        nr_shares += int(remaining_capital / data["price." + ticker].iloc[j])
                                        file1.write("Baught %d shares at %f price<=%f, total nr shares is %d" %
                                                    (int(remaining_capital / data["price." + ticker].iloc[j]),
                                                     data["price." + ticker].iloc[j],
                                                     indiff_price, nr_shares))
                                        print("Baught %d shares at %f price<=%f, total nr shares is %d" %
                                              (int(remaining_capital / data["price." + ticker].iloc[j]),
                                               data["price." + ticker].iloc[j],
                                               indiff_price, nr_shares))
                                        remaining_capital -= data["price." + ticker].iloc[j] * int(
                                            remaining_capital / data["price." + ticker].iloc[j])
                                    else:
                                        profit_start += data["price." + ticker].iloc[j] * \
                                                        data["size." + ticker].iloc[j]
                                        nr_shares += data["size." + ticker].iloc[j]
                                        file1.write("Baught %d shares at %f price <= %f, total nr shares is %d \n" %
                                                    (
                                                        data["size." + ticker].iloc[j], data["price." + ticker].iloc[j],
                                                        indiff_price,
                                                        nr_shares))
                                        print("Baught %d shares at %f price <= %f, total nr shares is %d" %
                                              (
                                                  data["size." + ticker].iloc[j], data["price." + ticker].iloc[j],
                                                  indiff_price,
                                                  nr_shares))
                                        remaining_capital -= data["price." + ticker].iloc[j] * \
                                                             data["size." + ticker].iloc[
                                                                 j]
                                    j += 1
                                profit_start = capital - remaining_capital

                            profit_end = data["price." + ticker].iloc[exit_index[i]] * data["size." + ticker].iloc[
                                exit_index[i]]
                            exit_prices.append(profit_end)
                            if exit_index[i]<length_data-1:
                                time_end.append(data["seconds"].iloc[exit_index[i]+1])
                            else:
                                time_end.append(data["seconds"].iloc[exit_index[i]])
                            times.append(time_end[-1] - time_start[-1])
                            profit_end = 0
                            j = exit_index[i]
                            if j < len_test - 2:
                                while nr_shares > 0:
                                    if data["size." + ticker].iloc[j] < nr_shares:
                                        profit_end += data["price." + ticker].iloc[j] * data["size." + ticker].iloc[
                                            j]
                                        nr_shares -= data["size." + ticker].iloc[j]
                                        file1.write("Sold %d shares at %f price, total nr shares is %d \n" %
                                                    (data["size." + ticker].iloc[j], data["price." + ticker].iloc[j],
                                                     nr_shares))
                                        print("Sold %d shares at %f price, total nr shares is %d" %
                                              (data["size." + ticker].iloc[j], data["price." + ticker].iloc[j],
                                               nr_shares))
                                        remaining_capital += data["price." + ticker].iloc[j] * \
                                                             data["size." + ticker].iloc[
                                                                 j]
                                    else:
                                        profit_end += data["price." + ticker].iloc[j] * nr_shares
                                        file1.write("Sold last %d shares at %f price, total nr shares is %d \n" %
                                                    (nr_shares, data["price." + ticker].iloc[j], 0))
                                        print("Sold last %d shares at %f price, total nr shares is %d" %
                                              (nr_shares, data["price." + ticker].iloc[j], 0))
                                        remaining_capital += data["price." + ticker].iloc[j] * nr_shares
                                        nr_shares = 0
                                    j += 1
                            else:
                                profit_end += data["price." + ticker].iloc[j] * nr_shares
                                file1.write(
                                    "Reached end of csv, so dumped the last %d shares at %f price, total nr shares is %d \n" %
                                    (nr_shares, data["price." + ticker].iloc[j], 0))
                                print(
                                    "Reached end of csv, so dumped the last %d shares at %f price, total nr shares is %d" %
                                    (nr_shares, data["price." + ticker].iloc[j], 0))
                                nr_shares = 0

                            profits.append(profit_end - profit_start)
                            returns.append((profit_end - profit_start) / profit_start)
                            nr_trades.append(exit_index[i] - entry_index[i])
                            count_trades += 1
                            row_labels.append("%s.trade%d" % (ticker, count_trades))
                            file1.write("Return is %f \n" % (profits[-1]/remaining_capital))
                            print("Return is %f" % (profits[-1]/remaining_capital))
                            file1.write("Profits are %f \n"%(profits[-1]))
                            print("Profits are %f \n"%(profits[-1]))

                        i += 1

    file1.write(
        "[%s] Testing completion %.2f%%, Time elapsed:%s, Testing loss: %.7f, Accuracy:%.4f%%, Overall Accuracy:%.4f%% \n"
        % (ticker, 100 * round(count_completion / nr_testing_csv, 4), timming(start), test_losses[-1],
           100 * round(accuracies[-1], 6), 100 * round(np.mean(accuracies), 6)))
    print(
        "[%s] Testing completion %.2f%%, Time elapsed:%s, Testing loss: %.7f, Accuracy:%.4f%%, Overall Accuracy:%.4f%%"
        % (ticker, 100 * round(count_completion / nr_testing_csv, 4), timming(start), test_losses[-1],
           100 * round(accuracies[-1], 6), 100 * round(np.mean(accuracies), 6)))

profit_data = pd.DataFrame()

profit_data["stock.trade"] = row_labels
profit_data["profit"] = profits
profit_data["return"] = returns
profit_data["nr trades"] = nr_trades
profit_data["time in trade"] = times
profit_data["entry price"] = entry_prices
profit_data["exit price"] = exit_prices
profit_data["POSIX entry time"] = time_start
profit_data["POSIX exit time"] = time_end
profit_data["entry time"] = [datetime.datetime.fromtimestamp(t).strftime('%Y-%m-%d %H:%M:%S.%f') for t in
                             time_start]
profit_data["exit time"] = [datetime.datetime.fromtimestamp(t).strftime('%Y-%m-%d %H:%M:%S.%f') for t in time_end]

profit_data.to_csv(
    "./Profits/Profits Fwd LMU trade no events, fixed lr, accuracy %.4f%%,hidden %d,memory %d,theta %d,batch %d,window %d.csv" %
    (100 * np.mean(accuracies), hidden_size, memory_size, theta, batch_size, window))

if flag_error == 1:
    if length_data > window:
        scaled_data.to_csv("LMU no events: predicted events for stock %s.csv" % (ticker))

file1.write("Sum of all profits after forwardtesting on %s is %.f \n" % (fwd_folder, sum(profits)))
print("Sum of all profits after forwardtesting on %s is %.f" % (fwd_folder, sum(profits)))

file1.write("Sum of all returns after forwardtesting on %s is %.f \n" % (fwd_folder, sum(profits) / capital))
print("Sum of all returns after forwardtesting on %s is %.f" % (fwd_folder, sum(profits) / capital))


file1.close()
