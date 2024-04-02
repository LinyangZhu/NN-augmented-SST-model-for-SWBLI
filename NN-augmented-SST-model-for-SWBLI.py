
import os
import time
import torch
import torch.nn as nn
import numpy as np
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.utils.data as data


class Net(nn.Module):
    """
    DNN framework
    """
    def __init__(self, layer):
        super(Net, self).__init__()
        self.fc = nn.Sequential()
        for i in range(len(layer)-1):
            self.fc.add_module('fc{}'.format(i),
                               nn.Linear(layer[i], layer[i+1]))
            if i != (len(layer) - 2):
                # self.fc.add_module('bn{}'.format(i), nn.BatchNorm1d(layer[i+1]))
                self.fc.add_module('Tanh{}'.format(i), nn.Tanh())

    def forward(self, x):
        x = self.fc(x)
        return x


def write_to_file(filename, matrix, first_col_int):
    fid = open(filename, 'w')
    if first_col_int == 'Y':
        for index, vector in enumerate(matrix):
            fid.write('%d ' % (index + 1))
            for value in vector:
                fid.write(str(value) + ' ')
            fid.write('\n')

    elif first_col_int == 'N':
        for vector in matrix:
            for value in vector:
                fid.write(str(value) + ' ')
            fid.write('\n')
    else:
        exit(1)
    fid.close()


def save_params():
    params = Fcnn_model.state_dict()
    file_params = r'./parameter.dat'
    torch.save(params, file_params)
    with open(file_params, 'a') as file_object:
        file_object.seek(0)
        file_object.truncate()
        for key, value in params.items():
            if 'num_batches_tracked' in key:
                continue
            value = value.view(1, -1)
            (x_d, y_d) = value.size()
            for xd in range(x_d):
                for yd in range(y_d):
                    file_object.write('%e ' % value[xd, yd])
            file_object.write('\n')


def weight_init(net):
    if isinstance(net, nn.Linear):
        nn.init.kaiming_uniform_(net.weight, a=0, nonlinearity='leaky_relu')
        # nn.init.normal_(net.weight)
        nn.init.constant_(net.bias, 0)


def linear_normalization(x, min_value, max_value):
    dim = x.shape[1]
    x_min = np.zeros(dim)
    x_max = np.zeros(dim)
    for i in range(dim):
        x_min[i] = np.min(x[:, i], axis=0)
        x_max[i] = np.max(x[:, i], axis=0)
        x[:, i] = (x[:, i] - x_min[i]) / (x_max[i] - x_min[i]) * (max_value - min_value) + min_value
    return x, x_min, x_max


def linear_normalize_test_data(x, norm_min, norm_max, x_min, x_max):
    dim = x.shape[1]
    for i in range(dim):
        x[:, i] = (x[:, i] - x_min[i]) / (x_max[i] - x_min[i]) * (norm_max - norm_min) + norm_min
    return x, x_min, x_max


def reverse_linear_normalization(x, norm_min, norm_max, x_min, x_max):
    y = (x - norm_min)/(norm_max - norm_min)*(x_max - x_min) + x_min
    return y

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# set the random seed to reproduce the result
seed = np.random.randint(0,100,size=1)
fid = open('Random_seed.dat', 'w')
fid.write('%d \n' % (seed))
fid.close()
print('Random Seed is: ', seed)
torch.manual_seed(seed)
np.random.seed(seed)

feature_list = [9, 10, 11, 12, 13]   #  9, 10, 11, 12, 13
# import training data
feature_start_index = 9
file = np.loadtxt('data_train.dat')
np.random.shuffle(file)
Y = file[:, -1]
Y = Y.reshape(-1, 1)
label_train = Y.copy()
X = file[:, 0:-1]
y_train_tau = Y * X[:, 1].reshape(-1, 1) * X[:, 2].reshape(-1, 1)
y_train_tau = torch.from_numpy(y_train_tau).type(torch.FloatTensor)
y_train_tau = y_train_tau.squeeze()
train_cell = file[:, 0:feature_start_index].copy()

# import validation data
file_pre = np.loadtxt('data_valid.dat')
test_cell = file_pre[:, 0:feature_start_index].copy()
y_pre = file_pre[:, -1:]
y_pre = y_pre.reshape(-1, 1)
label_test = y_pre.copy()
x_pre = file_pre[:, feature_list]
feature_no = x_pre.shape[1]
y_test_tau = y_pre * file_pre[:, 1].reshape(-1, 1) * file_pre[:, 2].reshape(-1, 1)
y_test_tau = torch.from_numpy(y_test_tau).type(torch.FloatTensor)
y_test_tau = y_test_tau.squeeze()
test_cell = file_pre[:, 0:feature_start_index].copy()

# linearizaiton
norm_min = -1
norm_max = 1
X_feature, X_min, X_max = linear_normalization(X[:, feature_list], norm_min, norm_max)
Y, Y_min, Y_max = linear_normalization(Y, norm_min, norm_max)

y_pre = (y_pre - Y_min) / (Y_max - Y_min) * (norm_max - norm_min) + norm_min
fid = open('min_max.dat', 'w')
for i in range(feature_no):
    fid.write('%e %e \n' % (X_min[i], X_max[i]))
fid.write('%e %e ' % (Y_min, Y_max))
fid.close()

for i in range(feature_no):
    x_pre[:, i] = (x_pre[:, i] - X_min[i]) / (X_max[i] - X_min[i]) * (norm_max - norm_min) + norm_min

y_max_tensor = torch.from_numpy(Y_max).type(torch.FloatTensor)
y_min_tensor = torch.from_numpy(Y_min).type(torch.FloatTensor)
data_train = np.hstack((X_feature, Y))
data_train_bar = np.hstack((X[:, 0:feature_start_index], data_train))
data_train = torch.from_numpy(data_train).type(torch.FloatTensor)
data_train_bar = torch.from_numpy(data_train_bar).type(torch.FloatTensor)
x_train, y_train = data_train_bar[:, feature_start_index:-1], data_train_bar[:, -1]

data_test = np.hstack((x_pre, y_pre))
data_test = torch.from_numpy(data_test).type(torch.FloatTensor)

width = 8
Fcnn_model = Net([feature_no, width, width, width, 1])   # width,
para_count = count_parameters(Fcnn_model)
print('Total Learnable Parameters: {}'.format(para_count))
fid = open('Learnable_Parameters.dat', 'w')
fid.write('%d \n' % (para_count))
fid.close()

batch_size_bar = int(X.shape[0]/64)  #  
data_loader = data.DataLoader(dataset=data_train_bar, batch_size=batch_size_bar, shuffle=True, drop_last=True)

Fcnn_model.apply(weight_init)  # weight initializaiton
global_step = 4001
train_loss = 1e6
optimizer = torch.optim.Adamax(Fcnn_model.parameters(), lr=1e-2)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.96, patience=50, verbose=True,
                                           threshold=1e-1, threshold_mode='rel', cooldown=0, min_lr=2e-06, eps=1e-06)
      
loss_func = nn.MSELoss()
tau_weight = 1.e3  # constraint loss coefficient

def train(model, train_loader):
    model.train()
    batch_loss_base = 0
    regularization = 0
    batch_loss_tau = 0
    batch_loss = 0
    for batch in train_loader:
        batch_x_bar, batch_y = batch[:, 0:-1], batch[:, -1]
        mixing_length = batch_x_bar[:, 1]
        strain_rate = batch_x_bar[:, 2]
        batch_x = batch_x_bar[:, feature_start_index:]
        batch_out = model(batch_x)
        batch_out = batch_out.squeeze()
        batch_loss_base = loss_func(batch_out, batch_y)
        batch_loss = batch_loss_base.clone()

        # constraint loss
        batch_out = (batch_out + 1) / 2 * (y_max_tensor - y_min_tensor) + y_min_tensor  # 反归一化
        batch_y = (batch_y + 1) / 2 * (y_max_tensor - y_min_tensor) + y_min_tensor  # 反归一化
        batch_tau_ml = strain_rate * batch_out * mixing_length
        batch_tau_label = strain_rate * batch_y * mixing_length
        batch_loss_tau = tau_weight*loss_func(batch_tau_ml, batch_tau_label)
        batch_loss += batch_loss_tau

        # regularization
        regularization_l1, regularization_l2 = 0, 0
        for param in model.parameters():
            regularization_l1 += torch.norm(param, 1)
            regularization_l2 += torch.norm(param, 2)

        regularization_l1 = 2e-5 * regularization_l1
        regularization_l2 = 1e-4 * regularization_l2
        regularization = regularization_l1 + regularization_l2
        batch_loss += regularization

        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
    scheduler.step(batch_loss)   
    return batch_loss_base.item(), regularization.item(), regularization_l1.item(), \
           regularization_l2.item(), batch_loss_tau.item()


def test(model, test_loader):
    model.eval()
    x_test = test_loader[:, 0:-1]
    y_test = test_loader[:, -1]
    out_test = model(x_test)
    out_test = out_test.squeeze()
    y_test = y_test.squeeze()
    test_loss = loss_func(out_test, y_test)
    return out_test, test_loss


plt.ion()
ax = []
loss_train_log = []
loss_test_log = []
plt.figure(figsize=(8, 5))

# save evolution of loss function
f1 = open('loss_component.dat', 'w')
f1.write('variables = "Epoch", "MSE loss", "Regulation loss", "Constraint loss"' + '\n')
f2 = open('loss_evolution.dat', 'w')
f2.write('variables = "Epoch", "Training loss", "Validation loss"' + '\n')
min_test_error = 1e6
optimal_epoch = 1
#  time-in
start_time = time.time()
for epoch in range(global_step):
    loss_base, loss_regularization, regularization_l1, regularization_l2, loss_constraint \
        = train(Fcnn_model, data_loader)
    if epoch % 10 == 0:
        print('MSE=', format(loss_base, '.2e'), 'L1-Norm=', format(regularization_l1, '.2e'),
              'L2-Norm=', format(regularization_l2, '.2e'), 'Constrain loss=', format(loss_constraint, '.2e'))
        f1.write(str(epoch + 1) + '  ' + str(loss_base) + '  ' + str(loss_regularization) + '  '
                 + str(loss_constraint) + '\n')
    if epoch % 10 == 0:
        # calculate test loss
        test_out, loss_test = test(Fcnn_model, data_test)
        # reverse normalization
        test_out = test_out.detach().cpu().numpy()
        test_out = reverse_linear_normalization(test_out, norm_min, norm_max, Y_min, Y_max)
        # the constraint calculated by model
        test_out_tau = test_out * file_pre[:, 2] * file_pre[:, 1]
        test_out_tau = torch.from_numpy(test_out_tau).type(torch.FloatTensor)
        loss_test += tau_weight * loss_func(test_out_tau, y_test_tau)
        loss_test = loss_test.item()
        # calculate train loss
        train_out, loss_train = test(Fcnn_model, data_train)
        # reverse normalization
        train_out = train_out.detach().cpu().numpy()
        train_out = reverse_linear_normalization(train_out, norm_min, norm_max, Y_min, Y_max)
        # the shear stress calculated by model
        train_out_tau = train_out * X[:, 2] * X[:, 1]
        train_out_tau = torch.from_numpy(train_out_tau).type(torch.FloatTensor)
        loss_train += tau_weight * loss_func(train_out_tau, y_train_tau)
        loss_train = loss_train.item()

        print('epoch=', epoch, 'Training loss=', format(loss_train, '.2e'),
              'Validation loss=', format(loss_test, '.2e'))
        f2.write(str(epoch + 1) + '  ' + str(loss_train) + '  ' + str(loss_test) + '\n')
        # plot the loss 
        ax.append(epoch+1)
        loss_train_log.append(loss_train)
        loss_test_log.append(loss_test)
        plt.clf()
        plt.axes([0.15, 0.15, 0.8, 0.8])
        plt.yscale('log')
        plt.plot(ax, loss_train_log, color='r', label='Training loss')
        plt.plot(ax, loss_test_log, color='b', label='Validation loss')
        plt.xlabel('Epoch', fontsize=16)
        plt.ylabel('Loss', fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.legend(fontsize=14)
        plt.pause(0.01)
        plt.ioff()

        # save the model according to its performance in validation dataset
        total_loss = loss_test  
        if epoch - optimal_epoch > 500:
            break
        if total_loss < min_test_error:
            print('---------------Refresh model !!!---------------', '\n')
            optimal_epoch = epoch
            min_test_error = total_loss
            save_params()
            # Saving & Loading a General Checkpoint for Inference and/or Resuming Training
            model_path = './Fcnn_model.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': Fcnn_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_func,
            }, model_path)
            # save predicted training data
            model_train = train_out.reshape(-1, 1)
            train_compare = np.hstack((label_train, model_train))
            train_compare = np.c_[train_cell, train_compare]
            file_name = r'./train_data.dat'
            write_to_file(file_name, train_compare, 'Y')
            # save predicted validation data
            model_test = test_out.reshape(-1, 1)
            test_compare = np.hstack((label_test, model_test))
            test_compare = np.c_[test_cell, test_compare]
            file_name = r'./test_data.dat'
            write_to_file(file_name, test_compare, 'Y')

f1.close()
f2.close()
# print the elapsed time
elapsed_time = time.time() - start_time
print('computation time = ', elapsed_time)
fid = open('Computation_time.dat', 'w')
fid.write('%e \n' % (elapsed_time))
fid.close()

plt.savefig('./Normal_training.jpg', dpi=600)
