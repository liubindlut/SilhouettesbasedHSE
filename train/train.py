import torch
import torch.nn as nn
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from data_processing import process_training_data, MyDataset, to_gpu, to_tensor, save_obj_mesh
from regressCNN import RegressionPCA
#from regressCNN_nonExchange import RegressionPCA
#from regressCNN_maxpooling import RegressionPCA

def shape_error(merge_shape, true_shape):
    error = torch.mean(torch.mean(torch.pow((merge_shape-true_shape), 2), 1))
    return error

def main():

    cuda = True
    device = torch.device('cuda:0' if cuda else 'cpu')

    gender = 'female'
    is_shuffle = True
    num_points = 972

    # batch_size = 128
    # learning_rate = 1e-5
    # training_epoch = 500

    batch_size = 32
    learning_rate = 1e-5
    training_epoch = 20

    front_train, side_train, shape_train = process_training_data(gender, is_shuffle, num_points)
    shape_train_pca = shape_train[:, 1:]

    data_train = MyDataset(front_train, side_train, shape_train_pca)
    data_loader = DataLoader(dataset = data_train, batch_size = batch_size, shuffle = True)
    train_len_bs = len(data_loader)
    print('==> num of batch:', train_len_bs)

    len_out = shape_train_pca.shape[1]
    print('==> the dimension of pca coeff:', len_out)
    model = RegressionPCA(len_out).to(device)
    print("==> Total parameters: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, betas=(0.9, 0.999))

    fig_loss_train = []
    fig_epoch = []
    for epoch in range(training_epoch):
        loss_n = 0
        for i, data_temp in enumerate(data_loader):
            # read data from data_loader, get 32 data each time
            front_e_ith, side_e_ith, shape_ith = data_temp
            front_e_ith, side_e_ith, shape_ith = \
                Variable(front_e_ith).to(device, dtype = torch.float), \
                    Variable(side_e_ith).to(device, dtype = torch.float), \
                        Variable(shape_ith).to(device, dtype = torch.float)
            # feed data and forward pass
            outputs = model(front_e_ith, side_e_ith)
            #********************************************************************************************
            loss = criterion(outputs, shape_ith.float())
            #print(loss.item())
            #temp_e = shape_error(outputs, shape_ith.float())
            #print(temp_e)
            #loss_n += temp_e
            loss_n += loss.item()
            #********************************************************************************************
            # backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # if epoch%5 == 0:
        #     print('epoch:{}  TrainLoss:{:.6f}'.format(epoch, loss_n))
        #     fig_loss_train = fig_loss_train + [loss_n]
        #     fig_epoch = fig_epoch + [epoch]
        #     torch.save(model.state_dict(), './model/model_{}_{}.ckpt'.format(gender, epoch))
        print('epoch:{}  TrainLoss:{:.6f}'.format(epoch, loss_n))
        fig_loss_train = fig_loss_train + [loss_n]
        fig_epoch = fig_epoch + [epoch]
        torch.save(model.state_dict(), './model/model_{}_{}.ckpt'.format(gender, epoch))

    np.save('./data/output/{}_fig_loss_train.npy'.format(gender), np.array(fig_loss_train))

    plt.plot(fig_epoch, fig_loss_train)
    plt.title('training loss')
    plt.show()


if __name__ == "__main__":
    main()