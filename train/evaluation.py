import torch
import torch.nn as nn
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from data_processing import process_training_data, process_test_data, MyDataset, to_gpu, to_tensor
from data_processing import save_obj_mesh, error_histogram_train, error_histogram_test, plot_error_histogram
from regressCNN import RegressionPCA
#from regressCNN_nonExchange import RegressionPCA
#from regressCNN_maxpooling import RegressionPCA

def main(gender, num_points):

    is_shuffle = False
    out_predict_pathFile = './data/output/female_TrainPredict/'
    out_predict_pathFile_test = './data/output/female_TestPredict/'

    front_train, side_train, shape_train = process_training_data(gender, is_shuffle, num_points)
    shape_train_pca = shape_train[:, 1:]
    ratioY_train = shape_train[:, 0]
    
    len_out = shape_train_pca.shape[1]
    fig_loss_train = np.load('./data/output/female_fig_loss_train.npy')
    min_index = np.argmin(fig_loss_train)
    print('==> the ith model was choosen:', min_index*5)
    modelName = './model/model_{}_{}.ckpt'.format(gender, 485)#min_index*5)#485)#
    #ourModel = RegressionPCA(len_out)
    #ourModel.load_state_dict(torch.load(modelName))
    #ourModel.eval()
    #save_obj_mesh(gender, out_predict_pathFile, ourModel, front_train, side_train, ratioY_train)
    #error_histogram_train(gender)

    front_test, side_test, ratioY_test = process_test_data(gender, num_points)
    ourModel_test = RegressionPCA(len_out)
    ourModel_test.load_state_dict(torch.load(modelName))
    ourModel_test.eval()
    save_obj_mesh(gender, out_predict_pathFile_test, ourModel_test, front_test, side_test, ratioY_test)
    error_histogram_test(gender)

    #plot_error_histogram()



if __name__ == "__main__":
    gender = 'female'
    num_points = 486
    main(gender, num_points)
    #plot_error_histogram()
