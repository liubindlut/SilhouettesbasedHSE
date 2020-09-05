import numpy as np
import trimesh
import cv2
import math
import codecs
import json
import torch

#-----------------------------------------
#-------- Model basis func
#----------------------------------------- 

def to_tensor(x):
    return torch.from_numpy(x).float()   

def repeat_data(x):
    num, row, col = x.shape
    data = np.zeros([num, row, col+2])
    data[:, :, 0] = x[:, :, -1]
    data[:, :, 1:-1] = x
    data[:, :, -1] = x[:, :, 0]
    return data

def getBinaryimage(img_dir_filename, resolution_normalization_height=600):
    #description: a white object and black background binary image is saved.

        img = cv2.imread(img_dir_filename)
        grayimage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, grayimage = cv2.threshold(grayimage, 230, 255, cv2.THRESH_BINARY)
        #ret, grayimage = cv2.threshold(grayimage, 0.5, 255, cv2.THRESH_BINARY)
        img[:,:,0] = grayimage
        img[:,:,1] = grayimage
        img[:,:,2] = grayimage
    
        #normalization height
        img_sum_row = np.sum(img[:,:,0], axis = 1)
        img_height = img_sum_row[img_sum_row>0]
        body_height = len(img_height)
        ratio = resolution_normalization_height/body_height
        rows, cols, channels = img.shape
        rows_temp = int(rows*ratio)
        cols_temp = int(cols*ratio)
        img_resize = cv2.resize(img, (cols_temp, rows_temp))

        return img_resize

def getSamplePoints(im, sample_num, m):
    # finding contours: white object from black background       
        imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        contours, hierarchy = cv2.findContours(imgray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        
        # choose a bigger aera
        index_i = 0
        max_area = -100
        for i in np.arange(len(contours)):
            cnt = contours[i]
            if cv2.contourArea(cnt) > max_area:
                max_area = cv2.contourArea(cnt)
                index_i = i
        num, rows, cols = contours[index_i].shape
        points = contours[index_i].reshape([num, cols])
    
        ind = np.linspace(0, num, sample_num, endpoint=False, dtype=int)
        sample_points = points[ind, :]
    
        return sample_points

def pca_reconstruction(pcaBase, pcaCoeff, mean_sample, ratio, filename):
    """reconstruction mesh as obj file
    
    parameters
    ----------
    pcaBase: eigenvector of covariance matrix
        num_feature * k
    pcaCoeff: the coefficient of base space
        1 * k
    mean_sample: average of all samples
        1 * num_feature 
    ratio: body height (pca perform on nomalization body height, 2 meters)
        float (meter)
    filename: write to file, include path 
    gender : female or male

    return
    --------
    nn output parameters, results are written into file
    """


    mesh = trimesh.load_mesh('./Models/PCA/averHuman.obj', process = False)
    vertices_t = mesh.vertices
    num_vertices, num_dim = vertices_t.shape[0], vertices_t.shape[1]
    averX = mean_sample.reshape((num_vertices, num_dim), order = 'F')
    pcaRecon = np.matmul(pcaCoeff, pcaBase.T)
    recon_sample = pcaRecon.reshape((num_vertices, num_dim), order = 'F')
    recon_points = ratio*(recon_sample + averX)
    mesh.vertices = recon_points
    mesh.export(filename)

def save_obj(filename, ourModel, body_height, data):

    #print('---------------begin to save OBJ file----------------')
    pcaBase = np.load('./Models/PCA/data_pcaBase.npy')
    mean_sample = np.load('./Models/PCA/data_mean_sample.npy')
    with torch.no_grad():
        front_train,  side_train = to_tensor(data[0]), to_tensor(data[1])
        front_train, side_train  = torch.unsqueeze(front_train, 0), torch.unsqueeze(side_train, 0)
        output = ourModel(front_train, side_train)
        output = output.numpy() # to numpy
        ratioY = body_height
        pcaCoeff = output
        pca_reconstruction(pcaBase, pcaCoeff, mean_sample, ratioY, filename)
    print('==> finished.')

# def modify():
#     img_filenames = ['./Images/5front.jpg', './Images/5side.jpg']
#     img = cv2.imread(img_filenames[1])
#     cv2.imshow("yuan", img)
#     cv2.flip(img,1)


