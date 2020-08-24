import numpy as np
import torch
import time
from torch.utils.data import Dataset, DataLoader
from widgetsMesh import write_objfile, reomve_files, datasets_to_datasets_error

import matplotlib.pyplot as plt

def process_training_data(gender, is_shuffle, num_points):
    """ adaptive to the input form of pytorch.

    parameters
    -------------
    gender: female or male
    is_shuffle: whether shuffle data
    num_points: the number of points in sampling silhouette 

    return
    ------------
    front_train, side_train, shape_train

    """
    data_front_file = './data/input/{}/{}/points_{}_frontview_linear.npy'.format(gender, num_points, gender)
    data_side_file = './data/input/{}/{}/points_{}_sideview_linear.npy'.format(gender, num_points, gender)
    data_shape_file = './data/input/{}/PCA/data_{}_trainShape.npy'.format(gender, gender)

    data_front = np.load(data_front_file)
    data_side = np.load(data_side_file)
    data_shape = np.load(data_shape_file)

    data_front = data_front.transpose(0, 2, 1)
    data_side = data_side.transpose(0, 2, 1)
    data_front = repeat_data(data_front)
    data_side = repeat_data(data_side)
    print('==> training shape label:', data_shape.shape)
    print('==> training 2 views input:', data_front.shape)

    num_sample = data_front.shape[0]
    if is_shuffle:
        shuffle_index = [index for index in range(num_sample)]
        np.random.shuffle(shuffle_index)
        data_front = data_front[shuffle_index]
        data_side = data_side[shuffle_index]
        data_shape = data_shape[shuffle_index]   

    front_train, side_train, shape_train = data_front, data_side, data_shape

    return front_train, side_train, shape_train

def process_test_data(gender, num_points):
    """ adaptive to the input form of pytorch.

    parameters
    -------------
    gender: female or male
    num_points: the number of points in sampling silhouette 

    return
    ------------
    front_test, side_test, shape

    """
    data_front_file = './data/input/{}/{}/test/points_{}_frontview_linear.npy'.format(gender, num_points, gender)
    data_side_file = './data/input/{}/{}/test/points_{}_sideview_linear.npy'.format(gender, num_points, gender)
    data_ratioY_file = './data/input/{}/PCA/data_{}_testratioY.npy'.format(gender, gender)

    data_front = np.load(data_front_file)
    data_side = np.load(data_side_file)
    data_ratioY = np.load(data_ratioY_file)

    data_front = data_front.transpose(0, 2, 1)
    data_side = data_side.transpose(0, 2, 1)
    data_front = repeat_data(data_front)
    data_side = repeat_data(data_side)
    print('==> test input', data_front.shape)

    front_test, side_test, ratioY_test = data_front, data_side, data_ratioY

    return front_test, side_test, ratioY_test

class MyDataset(Dataset):
    """ define __len__ and __getitem__ function explicitly
    function
    ------------
    __init__: initialize the parameters in MyDataset
    __len__ and __getitem__: call the parameters in __init__ function
    """
    def __init__(self, front_e_train, side_e_train, shape_train):
    #def __init__(self, front_train, front_e_train, side_train, side_e_train, shape_train):
        self.front_e_train = front_e_train
        self.side_e_train = side_e_train
        self.shape_train = shape_train

    def __len__(self):
        num = self.shape_train.shape[0]
        return num

    def __getitem__(self, index):
        front_e_ith = self.front_e_train[index]
        side_e_ith = self.side_e_train[index]
        shape_ith = self.shape_train[index]
        return front_e_ith, side_e_ith, shape_ith

def to_tensor(x):
    return torch.from_numpy(x).float()
def to_gpu(x, device):
    return x.to(device, dtype = torch.float)

def repeat_data(x):
    num, row, col = x.shape
    data = np.zeros([num, row, col+2])
    data[:, :, 0] = x[:, :, -1]
    data[:, :, 1:-1] = x
    data[:, :, -1] = x[:, :, 0]
    return data

def save_obj_mesh(gender, pathFile, ourModel, front_train, side_train, ratioY_train):
    """ save predict results into obj files.
    Note, front_train and side_train is not shuffled.
    parameters:
    --------------
    gender: female or male
    pathFile: output path
    ourModel: learned CNN model
    front_train: the point sets of front view 
    side_train: the point sets of side view
    shape_train: only need the body height parameters

    output:
    ------------
    all in pathFile

    """
    reomve_files(pathFile)
    print('---------------begin to save OBJ file----------------')
    pcaBase = np.load('./data/input/{}/PCA/data_{}_pcaBase.npy'.format(gender, gender))
    mean_sample = np.load('./data/input/{}/PCA/data_{}_mean_sample.npy'.format(gender, gender)) 
    n_train = front_train.shape[0]
    print('number of samples:', n_train)
    #---------------- save resutls --------------
    time_start = time.time()
    with torch.no_grad():
        front_train, side_train = to_tensor(front_train), to_tensor(side_train)
        output = ourModel(front_train, side_train)
        output = output.numpy() # to numpy
        for i in np.arange(n_train):
            filename_temp = '{}.obj'.format(i)
            filename = pathFile + filename_temp
            ratioY = ratioY_train[i]
            pcaCoeff = output[i]
            #print(np.linalg.norm(ratioY_train[i] - pcaCoeff))
            write_objfile(pcaBase, pcaCoeff, mean_sample, ratioY, filename, gender)

    time_end = time.time()
    print('Average time:', (time_end - time_start)/n_train)
    print(str(time_end - time_start))
    print('--------------------------------------------------------')

def error_histogram_train(gender):
    if gender == 'female':
        predPath = './data/output/female_TrainPredict/'
        TruePath = './data/input/female_TrainTrue/'
    else:
        predPath = './data/output/male_TrainPredict/'
        TruePath = './data/input/male_TrainTrue/'    

    error_list_aver1, max_aver1, mean_aver1, filename_max_aver1, error_list_max1, max_max1, mean_max1, filename_max_max1 \
            = datasets_to_datasets_error(gender, predPath, TruePath)

    np.save('./data/output/our_aver_list_train.npy', error_list_aver1)
    np.save('./data/output/our_max_list_train.npy', error_list_max1)
    print('the biggest aver error sample in training datas', filename_max_aver1)
    print('the biggest max error sample in training datas', filename_max_max1)
    print('Number of Train datasets:{}  max_maxError:{:.3f}  max_averError:{:.3f}  aver_maxError:{:.3f}  aver_averError:{:.3f}'\
        .format(len(error_list_aver1), 100*max_max1, 100*max_aver1, 100*mean_max1, 100*mean_aver1))

    return error_list_aver1, error_list_max1
    
def error_histogram_test(gender):
    if gender == 'female':
        predPath = './data/output/female_TestPredict/'
        TruePath = './data/input/female_TestTrue/'
    else:
        predPath = './data/output/male_TestPredict/'
        TruePath = './data/input/male_TestTrue/'  

    error_list_aver2, max_aver2, mean_aver2, filename_max_aver2, error_list_max2, max_max2, mean_max2, filename_max_max2 \
            = datasets_to_datasets_error(gender, predPath, TruePath)
    np.save('./data/output/our_aver_list_test.npy', error_list_aver2)
    np.save('./data/output/our_max_list_test.npy', error_list_max2)
    print('the biggest aver error sample in testing datas', filename_max_aver2)
    print('the biggest max error sample in testing datas', filename_max_max2)
    print('Number of Test datasets:{}  max_maxError:{:.3f}  max_averError:{:.3f}  aver_maxError:{:.3f}  aver_averError:{:.3f}'\
        .format(len(error_list_aver2), 100*max_max2, 100*max_aver2, 100*mean_max2, 100*mean_aver2))

    return error_list_aver2, error_list_max2

def draw_hist(listdata, title, position):
    plt.subplot(2, 2, position)
    plt.hist(listdata, bins=15, density=False, color = 'steelblue', edgecolor = 'k')
    plt.tick_params(top=False,right=False)
    plt.title(title)

def plot_error_histogram():

    error_list_aver1 = np.load('./data/output/our_aver_list_train.npy')
    error_list_max1 = np.load('./data/output/our_max_list_train.npy')
    error_list_aver2 = np.load('./data/output/our_aver_list_test.npy')
    error_list_max2 = np.load('./data/output/our_max_list_test.npy')

    plt.style.use('ggplot')
    draw_hist(100*np.asarray(error_list_aver1), 'training_mean_distribution', 1)
    draw_hist(100*np.asarray(error_list_max1), 'training_max_distribution', 2)
    draw_hist(100*np.asarray(error_list_aver2), 'testing_mean_distribution', 3)
    draw_hist(100*np.asarray(error_list_max2), 'testing_max_distribution', 4)
    plt.show()

if __name__ == "__main__":
    gender = 'female'
    is_shuffle = True
    num_points = 648
    process_training_data(gender, is_shuffle, num_points)