import os
import numpy as np
import glob
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from datasets.KNNGraph import KNNGraph
from datasets.AuxFunction import FFT
import pickle
# ------------------------------------------------------------
# signal size alse same as sample length . Both are dimension of the node
signal_size = 5

# label
label = [i for i in range(2)]
mean = []
std= []


# generate Training Dataset and Testing Dataset
def get_files (sample_length, root, InputType, task, overlapping_number,file_name,test=False):
    '''
    This function is used to generate the final training set and test set.
    root:The location of the data set
    normalname:List of normal data
    dataname:List of failure data
    '''
    validaition_data = []
    training_data = []
    label = 0
    normal = "_0"
    paths = glob.glob("data/SKABAcc2/{}/*.csv".format(file_name))
    print("paths",paths)
    # paths = [p for p in paths if p.find("acc2")>0]
    print("overlapping_number",overlapping_number)
    for j,path in enumerate(paths):
        
        if normal  in os.path.basename(path):
            data = data_load(sample_length,filename=path, label=label,InputType=InputType,task=task,overlapping_number = overlapping_number )
        elif '_1'  in os.path.basename(path):
            data = data_load(sample_length,filename=path, label= 1,InputType=InputType,task=task,overlapping_number = overlapping_number )
        else:
            data = data_load(sample_length,filename=path, label=j,InputType=InputType,task=task,overlapping_number = overlapping_number )
        if j == 0:
            training_data = data
        # print('Number of generated graphs in the following path('+path+')->'+len(data1))
        else:
            print("yes",j)
            validaition_data += data

    return training_data,validaition_data

# generate graohs which have overlapping nodes
def get_overlapping_subarrays(arr, window_size, stride):
    """
    Returns a list of overlapping subarrays of size window_size from the given array.

    Args:
        arr (list): The input array.
        window_size (int): The size of the sliding window.
        stride (int): The stride for sliding the window.

    Returns:
        A list of overlapping subarrays.
    """
    arr = arr.transpose()
    result = []
    for i in range(0, len(arr) - window_size + 1, stride):
        result.append(arr[i:i+window_size])
    return result

# Define a function to normalize a column of the dataframe
def normalize_col(col):
    col_numeric = pd.to_numeric(col, errors='coerce')
    col_normalized = (col_numeric - col_numeric.min()) / (col_numeric.max() - col_numeric.min())
    return col_normalized

def z_score_normalize_columns_for_testing_data(data, means, stds):

    # Subtract the mean and divide by the standard deviation for each column
    normalized_data = (data - means) / stds

    return normalized_data

def z_score_normalize_columns(data):
    """
    Perform z-score normalization on the columns of the given dataset.

    Parameters:
    data (numpy array): The dataset to be normalized.

    Returns:
    numpy array: The normalized dataset.
    """

    # Compute the mean and standard deviation of each column
    means = np.mean(data, axis=0)
    stds = np.std(data, axis=0)

    # Subtract the mean and divide by the standard deviation for each column
    normalized_data = (data - means) / stds

    return normalized_data, means, stds

def data_load(signal_size,filename, label, InputType, task, overlapping_number):
    '''
    This function is mainly used to generate test data and training data.
    filename:Data location
    axisname:Select which channel's data,---->"_DE_time","_FE_time","_BA_time"
    '''
    global mean,std
    fl = pd.read_csv(filename, header=None)
    if 'anomaly_free'  in os.path.basename(filename):
        print("filename",filename)
        fl,mean, std = z_score_normalize_columns(fl)
    else:
        fl = z_score_normalize_columns_for_testing_data(fl,mean,std)

    fl = fl.values
    fl = fl.reshape(-1,)
    data = []
    max_length =(fl.shape[0] // signal_size) * signal_size
    start, end = 0, signal_size
    while end <= fl[:max_length].shape[0]:
        if InputType == "TD":
            x = fl[start:end]
        elif InputType == "FD":
            x = fl[start:end]
            x = FFT(x)
        else:
            print("The InputType is wrong!!")

        data.append(x)
        start += signal_size
        end += signal_size

    graphset = KNNGraph(10,data,label,task)
    print("length of graphset",len(graphset),len(graphset[0]))
    return graphset




class SKABAcc2Knn(object):
    num_classes = 2


    def __init__(self, sample_length, data_dir,InputType,task,overlapping_number,file_name):
        self.sample_length = sample_length
        self.data_dir = data_dir
        self.InputType = InputType
        self.task = task
        self.overlapping_number = overlapping_number
        self.file_name = file_name


    def data_preprare(self, test=False):
        if len(os.path.basename(self.data_dir).split('.')) == 2:
            with open(self.data_dir, 'rb') as fo:
                list_data = pickle.load(fo, encoding='bytes')
        else:
            training_data, validataion_data = get_files(self.sample_length, self.data_dir, self.InputType, self.task, self.overlapping_number, self.file_name, test)
            with open(os.path.normpath(os.path.join('C:\\Users\\situser\\Desktop\\Vinothini\\SUTD\\PHMGNNBenchmark\\data\\SKABAcc2', "SKABAcc2Knn.pkl")), 'wb') as fo:
                print("datadir",self.data_dir)
                pickle.dump(training_data, fo)
                pickle.dump(validataion_data, fo)

        if test:
            test_dataset = list_data
            return test_dataset
        else:

            train_dataset, val_dataset = training_data, validataion_data
            print("Number of graphs in training dataset = ",len(train_dataset))
            print("Number of graphs in validating dataset = ",len(val_dataset))
            return train_dataset, val_dataset
