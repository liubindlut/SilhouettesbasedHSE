import trimesh
import numpy as np
from numpy.linalg import norm
import os
import json

def reomve_files(file_path):
    for name in os.listdir(file_path):
        file = os.path.join(file_path, name)
        if os.path.isfile(file):
            os.remove(file)
        else:
            reomve_files(file)

def write_objfile(pcaBase, pcaCoeff, mean_sample, ratio, filename, gender):
    """reconstruction mesh as obj file
    
    parameters
    ----------
    pcaBase: eigenvector of covariance matrix
        num_feature * k
    pcaCoeff: the coefficient of base space
        1 * k
    mean_sample: average of all samples
        1 * num_feature 
    ratio: body height (pca perform on nomalization body height, 1 meters)
        float (meter)
    filename: write to file, include path 
    gender : female or male

    return
    --------
    nn output parameters, results are written into file
    """

    if gender == 'female':
        mesh = trimesh.load_mesh('./data/input/female/PCA/averHuman.obj', process = False)
    else:
        mesh = trimesh.load_mesh('./data/input/male/PCA/averHuman.obj', process = False)

    vertices_t = mesh.vertices
    num_vertices, num_dim = vertices_t.shape[0], vertices_t.shape[1]
    averX = mean_sample.reshape((num_vertices, num_dim), order = 'F')
    pcaRecon = np.matmul(pcaCoeff, pcaBase.T)
    recon_sample = pcaRecon.reshape((num_vertices, num_dim), order = 'F')
    recon_points = ratio*(recon_sample + averX)
    mesh.vertices = recon_points
    mesh.export(filename)
    
def vertex_to_vertex_error(predobj, trueobj, null_vertex):
    """ calculate vertex to vertex error between two meshes

    parameters
    ----------
    predobj, trueobj: filename (including path)

    return
    ----------
    error_list: the error list among all vertex pairs
    """

    null_vertex = None
    trueMesh = trimesh.load_mesh(trueobj, process = False)
    estiMesh = trimesh.load_mesh(predobj, process  = False)

    points_true = trueMesh.vertices
    center_true = np.mean(points_true, axis=0)
    points_true = points_true - center_true

    points_esti = estiMesh.vertices
    center_esti = np.mean(points_esti, axis=0)
    points_esti = points_esti - center_esti

    error_list = norm(points_true-points_esti, ord=None, axis = 1)
    if null_vertex is not None:
        error_list[null_vertex-1] = 0.0
    return error_list

def datasets_to_datasets_error(gender, pred_path, true_path):
    """ calculate the maximum mean error and max error between two datasets

    parameters
    ----------
    pred_path, true_path: obj files are generated

    return
    ----------
    error_list_aver: the average error list of corresponding mesh in datasets
    error_list_max: the max error list of corresponding mesh in datasets
    """

    #null_vertex = np.loadtxt('./data/input/{}/nodes.txt'.format(gender))
    #null_vertex = null_vertex.astype(int)
    null_vertex = None

    filename = [name for name in os.listdir(pred_path) if os.path.join(pred_path, name)]
    errorList_aver = []
    errorList_max = []
    for name in filename:
        filename1 = pred_path + name
        filename2 = true_path + name
        error_list_vertex = vertex_to_vertex_error(filename1, filename2, null_vertex)
        error_mean, error_max = np.mean(error_list_vertex), np.max(error_list_vertex)
        errorList_aver = errorList_aver + [error_mean]
        errorList_max = errorList_max + [error_max]

    error_list_aver, error_list_max = np.asarray(errorList_aver), np.asarray(errorList_max)
    max_aver, mean_aver, filename_max_aver = np.max(error_list_aver), np.mean(error_list_aver), filename[np.argmax(error_list_aver)]
    max_max, mean_max, filename_max_max = np.max(error_list_max), np.mean(error_list_max), filename[np.argmax(error_list_max)]

    max_aver_index, max_max_index = np.argsort(error_list_aver)[-10:], np.argsort(error_list_max)[-10:]
    for i in max_aver_index:
        print(filename[i])
    print('----------------')
    for i in max_max_index:
        print(filename[i])

    return error_list_aver, max_aver, mean_aver, filename_max_aver, error_list_max, max_max, mean_max, filename_max_max  

def define_color_map(error_normalize):
    """ define color map based on vertex_error
    """
    vertex_color = np.zeros([len(error_normalize),4])
    for i in np.arange(len(error_normalize)):
        temp = error_normalize[i]
        if temp>0.75:
            red = 255
            green = np.floor(255*(1.0-(temp-0.75)/0.25))
            blue = 0
            vertex_color[i] = [red, green, blue, 255]
        elif temp>0.5:
            red = np.floor(255*(temp-0.5)/0.25)
            green = 255
            blue = 0
            vertex_color[i] = [red, green, blue, 255]
        elif temp>0.25:
            red = 0
            green = 255
            blue = np.floor(255*(1.0-(temp-0.25)/0.25))
            vertex_color[i] = [red, green, blue, 255]
        elif temp > -0.00001:
            red = 0
            green = np.floor(255*(temp/0.25))
            blue = 255
            vertex_color[i] = [red, green, blue, 255]  
    return vertex_color
    

def show_error_map(gender, predfile, truefile):

    #null_vertex = np.loadtxt('./data/input/{}/nodes.txt'.format(gender))
    #null_vertex = null_vertex.astype(int)
    null_vertex = []
    error_vlist = vertex_to_vertex_error(predfile, truefile, null_vertex)
    predMesh = trimesh.load_mesh(predfile, process = False)


    maxV = 0.01
    minV = 0.00001
    #maxV, minV = error_vlist.max(), error_vlist.min() # maxV and minV can be specified by datasets or other bad results
    #print('max error:{:.3f}, min error:{:.3f}'.format(maxV*100, minV*100))
    error_normalize = (error_vlist-minV)/(maxV-minV)

    # color_range = [[0, 255, 0, 255],[255, 0, 0, 255]]
    # vertex_color = trimesh.visual.color.linear_color_map(error_normalize, color_range=color_range)

    vertex_color = define_color_map(error_normalize)
    for j in null_vertex:
        vertex_color[j-1] = [240, 240, 240, 255]

    predMesh.visual.vertex_colors = vertex_color
    predMesh.show()
    # show overlap
    trueMesh = trimesh.load_mesh(truefile)
    trueMesh.visual.vertex_colors = [240, 240, 240, 255]
    scene = trimesh.Scene([predMesh, trueMesh])
    scene.show()

def show_overlap(objfile1, objfile2):
    mesh1 = trimesh.load_mesh(objfile1)
    mesh2 = trimesh.load_mesh(objfile2)
    mesh1.visual.face_colors = [0, 255, 155, 255]
    mesh2.visual.face_colors = [0, 155, 255, 255]
    scene = trimesh.Scene([mesh1, mesh2])
    scene.show()

def main():
    gender = 'female'
    pred_file = './data/output/female_Test/1854.obj' #  613, 698
    true_file = './data/output/female_True/1854.obj'
    show_error_map(gender, pred_file, true_file)




if __name__ == '__main__':
    main()