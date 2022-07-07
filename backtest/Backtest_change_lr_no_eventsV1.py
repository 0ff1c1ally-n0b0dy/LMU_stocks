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

torch.cuda.is_available()
ngpu = 1
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

###########################
#Create lists for names of folders for backtesting
#Declare low lr and high lr
##########################
folders_list=["Batch1","Batch2"]
nr_folders=len(folders_list)
low_lr=3*10**(-5)
#low_lr_initial=3*10**(-5)
high_lr=10**(-4)
#high_lr_initial=10**(-4)
warmup_steps=4000

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
#lr = 3e-5
criterion = nn.BCELoss()

optimizer_high_lr = torch.optim.Adam(model.parameters(), lr=high_lr, amsgrad=True)

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

file1_text = "./Prints/Output backtest no events, change lr: epochs %d, hidden %d,memory %d,theta %d,window %d.txt" % (
epochs, hidden_size, memory_size, theta, window)
file1 = open(file1_text, "w")
total_steps_low_lr=0
total_steps_high_lr=0
flag_error=1

for folder in folders_list:
    ###################################
    #First training pass with low learning rate
    #######################################
    print("Backtesting for %s started with low lr=%f"%(folder,low_lr))
    file1.write("Backtesting for %s started with low lr=%f"%(folder,low_lr))
    csvs = glob.glob("./"+folder+"/*.csv")
    nr_csv = len(csvs)
    nr_training_csv = int((1 - percentage_test) * nr_csv)
    training_csv = random.sample(range(nr_csv), nr_training_csv)
    testing_csv = [idx for idx in range(nr_csv) if idx not in training_csv]
    nr_testing_csv = len(testing_csv)

    epoch = 1
    while epoch <= epochs and flag_error == 1:
        running_loss = 0
        train_iters = 0
        train_losses = []
        for i in training_csv:
            start = time.time()
            file_name = csvs[i]
            ticker = file_name.split("/")[2].split(" ")[0]
            flag_error = 1
            count_training += 1
            try:
                data = pd.read_csv(file_name, low_memory=True,
                                   usecols=["seconds", "size." + ticker, "price." + ticker, "events." + ticker],
                                   dtype={"size." + ticker: "float32", "price." + ticker: "float32",
                                          "events." + ticker: "int8"})
            except ValueError:
                flag_error = 0
                file1.write("For %s the columns do not have the ticker's name \n" % (file_name.split("/")[2]))
                print("For %s the columns do not have the ticker's name" % (file_name.split("/")[2]))
            n_train = len(data)
            total_steps_low_lr+=epoch*n_train
            if n_train > 0 and flag_error == 1 and n_train - window > 0:
                step_num = total_steps_low_lr
                low_lr = (input_size ** (-0.5) )* min(step_num**(-0.5),step_num*(warmup_steps**(-1.5)))
                optimizer_low_lr = torch.optim.Adam(model.parameters(), lr=low_lr, amsgrad=True)
                file1.write("Training for %s in folder %s started with lr=%f \n" % (file_name.split("/")[2],folder,low_lr))
                print("Training for %s in folder %s started with lr=%f" % (file_name.split("/")[2],folder,low_lr))
                # data=pd.DataFrame(data,columns=["ask."+ticker,"ask_size."+ticker,"bid."+ticker,"bid_size."+ticker,"events."+ticker])
                scaled_data = data.copy()
                scaled_data=scaled_data[["seconds", "size." + ticker, "price." + ticker]]
                columns = list(scaled_data.columns)
                n_var = len(columns)
                for k in range(n_var):
                    scaler = MinMaxScaler()
                    scaled_data[columns[k]] = scaler.fit_transform(np.array(scaled_data[columns[k]]).reshape(-1, 1))

                for k in range(n_train - window):
                    model.zero_grad()
                    x_train = scaled_data.values[k:(k + window), :]
                    # x_train=X_train[k]
                    x_train = torch.FloatTensor(x_train).to(device)
                    if x_train.size(0) != 0:
                        x_train = x_train.view(batch_size, window, n_var)
                        y_train = data["events." + ticker].iloc[k + window]
                        y_train = torch.FloatTensor([y_train]).to(device)
                        y_pred = model(x_train)
                        if y_pred >= p_thr:
                            output_class = 1
                        else:
                            output_class = 0
                        loss = criterion(y_pred, y_train)
                        running_loss += loss.item()
                        loss.backward()
                        optimizer_low_lr.step()
                        train_iters += 1

                train_losses.append(running_loss / train_iters)
                file1.write("[%d/%d:%s/%s] Training completion:%.2f%%, Time elapsed:%s, Training loss: %.7f, lr=%f \n"
                            % (epoch, epochs, folder, ticker, 100 * round(count_training / (nr_training_csv * epochs*nr_folders), 4),
                               timming(start), running_loss / (train_iters),low_lr))

                # print("Training completion:%.2f%%"%(100*round(count_training/(nr_training_csv*epochs),4)))

                print("[%d/%d:%s/%s] Training completion:%.2f%%, Time elapsed:%s, Training loss: %.7f, lr=%f"
                      % (
                      epoch, epochs, folder, ticker, 100 * round(count_training / (nr_training_csv * epochs*nr_folders), 4), timming(start),
                      running_loss / (train_iters),low_lr))

        file1.write("[%d/%d:%s/%s] Training completed with low lr=%f, Average training loss: %.7f \n" % (
            epoch, epochs, folder, ticker, low_lr, np.mean(train_losses)))
        print("[%d/%d:%s/%s] Training completed with low lr=%f, Average training loss: %.7f" % (
        epoch, epochs, folder, ticker, low_lr, np.mean(train_losses)))

        #############################
        # Second training pass with high learning rate
        ################################
        file1.write("Backtesting for %s started with high lr=%f" % (folder, high_lr))


        running_loss = 0
        train_iters = 0
        train_losses = []
        for i in training_csv:
            start = time.time()
            file_name = csvs[i]
            ticker = file_name.split("/")[2].split(" ")[0]
            flag_error = 1
            count_training += 1
            try:
                data = pd.read_csv(file_name, low_memory=True,
                                       usecols=["seconds", "size." + ticker, "price." + ticker, "events." + ticker],
                                       dtype={"size." + ticker: "float32", "price." + ticker: "float32",
                                              "events." + ticker: "int8"})
            except ValueError:
                flag_error = 0
                file1.write("For %s the columns do not have the ticker's name \n" % (file_name.split("/")[2]))
                print("For %s the columns do not have the ticker's name" % (file_name.split("/")[2]))
            n_train = len(data)
            if n_train > 0 and flag_error == 1 and n_train - window > 0:
                total_steps_high_lr+=n_train*epoch
                step_num = total_steps_high_lr
                high_lr = (input_size ** (-0.5)) * min((step_num ** (-0.5)), step_num * (warmup_steps ** (-1.5)))
                optimizer_high_lr = torch.optim.Adam(model.parameters(), lr=high_lr, amsgrad=True)
                file1.write("Training for %s in folder %s started with lr=%f \n" % (
                file_name.split("/")[2], folder, high_lr))
                print(
                        "Training for %s in folder %s started with lr=%f" % (file_name.split("/")[2], folder, high_lr))
                    # data=pd.DataFrame(data,columns=["ask."+ticker,"ask_size."+ticker,"bid."+ticker,"bid_size."+ticker,"events."+ticker])
                scaled_data = data.copy()
                scaled_data = scaled_data[["seconds", "size." + ticker, "price." + ticker]]
                columns = list(scaled_data.columns)
                n_var = len(columns)
                for k in range(n_var):
                    scaler = MinMaxScaler()
                    scaled_data[columns[k]] = scaler.fit_transform(np.array(scaled_data[columns[k]]).reshape(-1, 1))

                for k in range(n_train - window):
                    model.zero_grad()
                    x_train = scaled_data.values[k:(k + window), :]
                    # x_train=X_train[k]
                    x_train = torch.FloatTensor(x_train).to(device)
                    if x_train.size(0) != 0:
                        x_train = x_train.view(batch_size, window, n_var)
                        y_train = data["events." + ticker].iloc[k + window]
                        y_train = torch.FloatTensor([y_train]).to(device)
                        y_pred = model(x_train)
                        if y_pred >= p_thr:
                            output_class = 1
                        else:
                            output_class = 0
                        loss = criterion(y_pred, y_train)
                        running_loss += loss.item()
                        loss.backward()
                        optimizer_high_lr.step()
                        train_iters += 1

                train_losses.append(running_loss / train_iters)
                file1.write(
                        "[%d/%d:%s/%s] Training completion:%.2f%%, Time elapsed:%s, Training loss: %.7f, lr=%f \n"
                        % (epoch, epochs, folder, ticker, 100 * round(count_training / (nr_training_csv * epochs * nr_folders), 4),
                           timming(start), running_loss / (train_iters), high_lr))

                    # print("Training completion:%.2f%%"%(100*round(count_training/(nr_training_csv*epochs),4)))

                print("[%d/%d:%s/%s] Training completion:%.2f%%, Time elapsed:%s, Training loss: %.7f, lr=%f"
                          % (
                              epoch, epochs, folder, ticker,
                              100 * round(count_training / (nr_training_csv * epochs*nr_folders), 4), timming(start),
                              running_loss / (train_iters), high_lr))

        file1.write("[%d/%d:%s/%s] Training completed with high lr=%f, Average training loss: %.7f \n" % (
                epoch, epochs, folder, ticker, high_lr, np.mean(train_losses)))
        print("[%d/%d:%s/%s] Training completed with high lr=%f, Average training loss: %.7f" % (
                epoch, epochs, folder, ticker, high_lr, np.mean(train_losses)))

        ##################
        #One evaluation: a weaker form of forwardtesting, where LMU is not optimized. We only do this once therefore.
        ###################

        test_losses = []
        test_loss = 0
        accuracies_per_epoch = []
        with torch.no_grad():
            for i in testing_csv:
                start = time.time()
                file_name = csvs[i]
                ticker = file_name.split("/")[2].split(" ")[0]
                flag_error = 1
                count_testing += 1
                try:
                    data = pd.read_csv(file_name, low_memory=True,
                                       usecols=["seconds", "size." + ticker, "price." + ticker, "events." + ticker],
                                       dtype={"size." + ticker: "float32", "price." + ticker: "float32",
                                              "events." + ticker: "int8"})
                except ValueError:
                    flag_error = 0
                    file1.write("For %s the columns do not have the ticker's name \n" % (file_name.split("/")[2]))
                    print("For %s the columns do not have the ticker's name" % (file_name.split("/")[2]))
                n_test = len(data)
                if n_test > 0 and flag_error == 1 and n_test - window > 0:
                    file1.write("Testing for %s from folder %s started \n" % (file_name.split("/")[2],folder))
                    print("Testing for %s started from folder %s started" % (file_name.split("/")[2],folder))
                    targets = []
                    outputs = []
                    accuracy = 0
                    test_iters = 0
                    # data=pd.DataFrame(data,columns=["ask."+ticker,"ask_size."+ticker,"bid."+ticker,"bid_size."+ticker,"events."+ticker])
                    scaled_data = data.copy()
                    scaled_data=scaled_data[["seconds", "size." + ticker, "price." + ticker]]
                    columns = list(scaled_data.columns)
                    n_var = len(columns)
                    for k in range(n_var):
                        scaler = MinMaxScaler()
                        scaled_data[columns[k]] = scaler.fit_transform(np.array(scaled_data[columns[k]]).reshape(-1, 1))
                    for j in range(n_test - window):
                        x_test = scaled_data.values[j:(j + window), :]
                        test_iters += 1
                        x_test = torch.FloatTensor(x_test).to(device)
                        if x_test.size(0) != 0:
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

                        if output_class == int(y_test):
                            accuracy += 1

                        outputs.append(output_class)

                    test_losses.append(test_loss / (test_iters))
                    accuracy = float(accuracy) / (test_iters)
                    accuracies.append(accuracy)
                    accuracies_per_epoch.append(accuracy)

                    file1.write(
                        "[%d/%d:%s/%s] Testing completion:%.2f%%, Time elapsed:%s, Testing loss: %.7f, Accuracy:%.4f%%, Accuracy/epoch:%.4f%%, Overall accuracy:%.4f%% \n"
                        % (
                        epoch, epochs, folder, ticker, 100 * round(count_testing / (nr_testing_csv * epochs*nr_folders), 4), timming(start),
                        test_loss / (test_iters), 100 * round(accuracy, 6),
                        100 * round(np.mean(accuracies_per_epoch), 6), 100 * round(np.mean(accuracies), 6)))

                    # print("Testing completion:%.2f%%"%(100*round(count_testing/(nr_testing_csv*epochs),4)))

                    print(
                        "[%d/%d:%s/%s] Testing completion:%.2f%%, Time elapsed:%s, Testing loss: %.7f, Accuracy:%.4f%%, Accuracy/epoch:%.4f%%, Overall accuracy:%.4f%%"
                        % (
                        epoch, epochs, folder, ticker, 100 * round(count_testing / (nr_testing_csv * epochs*nr_folders), 4), timming(start),
                        test_loss / (test_iters), 100 * round(accuracy, 6),
                        100 * round(np.mean(accuracies_per_epoch), 6), 100 * round(np.mean(accuracies), 6)))

        epoch += 1
        file1.write(
            "[%d/%d:%s/%s] Testing completed, Average testing loss: %.7f, Accuracy/epoch:%.4f%%, Overall accuracy:%.4f%% \n"
            % (epoch - 1, epochs, folder, ticker, np.mean(test_losses), 100 * round(np.mean(accuracies_per_epoch), 6),
               100 * round(np.mean(accuracies), 6)))

        print("[%d/%d:%s/%s] Testing completed, Average testing loss: %.7f, Accuracy/epoch:%.4f%%, Overall accuracy:%.4f%%"
              % (epoch - 1, epochs, folder, ticker, np.mean(test_losses), 100 * round(np.mean(accuracies_per_epoch), 6),
                 100 * round(np.mean(accuracies), 6)))

        torch.save(model,
               "./Saved_weights/"+folder+" LMU trade, b. no events change lr: epochs %d, accuracy %.4f%%,hidden %d,memory %d,theta %d,batch %d,window %d.pth" %
                (epoch, 100 * np.mean(accuracies_per_epoch), hidden_size, memory_size, theta, batch_size,window))

file1.close()
