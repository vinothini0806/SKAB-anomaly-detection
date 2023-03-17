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



# generate Training Dataset and Testing Dataset
def get_files(sample_length, root, InputType, task, overlapping_number, test=False):
    '''
    This function is used to generate the final training set and test set.
    root:The location of the data set
    normalname:List of normal data
    dataname:List of failure data
    '''
    data = []
    
    paths = glob.glob("data/SKABMD/*.csv")
    # paths = [p for p in paths if p.find("acc2")>0]
    print("overlapping_number",overlapping_number)
    for j,path in enumerate(paths):
        data1 = data_load(sample_length,filename=path, label=j,InputType=InputType,task=task,overlapping_number = overlapping_number )
        # print('Number of generated graphs in the following path('+path+')->'+len(data1))
        data += data1
    # This for loop is to get all 9 channel1 fault data as data
    # for i in tqdm(range(4)):
    #     print("i")
    #     print(i)
    #     i = i +1
    #     data_dir = 'valve'+ str(i)
    #     data_name = 'valve'+str(i)+'_acc1.csv'
    #     path2 = os.path.normpath(os.path.join('data\\SKAB',data_dir, data_name))
    #     # print("path2 ",root )
    #     data1 = data_load(sample_length,filename=path2, label=label[i],InputType=InputType,task=task)
    #     print("path2")
    #     print(path2)
    #     data += data1

    return data

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

def data_load(signal_size,filename, label, InputType, task, overlapping_number):
    '''
    This function is mainly used to generate test data and training data.
    filename:Data location
    axisname:Select which channel's data,---->"_DE_time","_FE_time","_BA_time"
    '''
    fl = pd.read_csv(filename, header=None)
    # normalizes the values in each column of fl between 0 and 1
    fl = fl.apply(normalize_col, axis=0)
    print("length of fl",len(fl))
    # fl = (fl - fl.min()) / (fl.max() - fl.min())
    fl = fl.values
    
    # fl = np.array(fl)
    # num_rows = fl.shape[0]
    # print("num_rows",num_rows)
    # fl = fl.reshape((num_rows, 8))
    # fl = fl.to_numpy()
    print("shape",fl.shape)
    # reshapes fl to a 1D array using the reshape() method.
    # fl = np.array(fl).reshape((-1, 8))
    # fl = fl.reshape(-1,)
    # print("Hello",len(fl))
    data = []
    # gives the maximum number of elements that can be used in the signals while ensuring that the signals are all of equal size
    max_length =(fl.shape[0] // signal_size) * signal_size
    start, end = 0, signal_size
    # print("overlapping_number", overlapping_number)
    # print("fl[:max_length].shape[0]",fl[:max_length].shape[0])
    if overlapping_number == 0:
        # print("HelloHello")
        while end <= fl[:max_length].shape[0]:
            if InputType == "TD":
                x = fl[start:end]
                x = x.transpose()
                # print("x",x)
            elif InputType == "FD":
                x = fl[start:end]
                x = FFT(x)
            else:
                print("The InputType is wrong!!")
            # List of nodes. here each nodes are list which contains the node features with lenth as signal_size
            data.append(x)
            start += signal_size
            end += signal_size
    elif overlapping_number > 0:
        # data.append(get_overlapping_subarrays(fl[:max_length], signal_size, overlapping_number))
        fl = fl.transpose()
        for row in fl:
            result =get_overlapping_subarrays(row, signal_size, overlapping_number) 
            # print(result)
            data.append(result)

        # Print the new array
        data = list(zip(*data))
        data = np.array(data)
        # print(data[1])
    else:
        print("The overapping number is invalid!!")


    graphset = KNNGraph(9,data,label,task)
    print("length of graphset",len(graphset),len(graphset[0]))
    return graphset



class SKABMDKnn(object):
    num_classes = 2


    def __init__(self, sample_length, data_dir,InputType,task,overlapping_number):
        self.sample_length = sample_length
        self.data_dir = data_dir
        self.InputType = InputType
        self.task = task
        self.overlapping_number = overlapping_number



    def data_preprare(self, test=False):
        if len(os.path.basename(self.data_dir).split('.')) == 2:
            with open(self.data_dir, 'rb') as fo:
                list_data = pickle.load(fo, encoding='bytes')
        else:
            list_data = get_files(self.sample_length, self.data_dir, self.InputType, self.task, self.overlapping_number, test)
            with open(os.path.normpath(os.path.join('C:\\Users\\situser\\Desktop\\Vinothini\\SUTD\\PHMGNNBenchmark\\data\\SKABMD', "SKABMDKnn.pkl")), 'wb') as fo:
                print("datadir",self.data_dir)
                pickle.dump(list_data, fo)

        if test:
            test_dataset = list_data
            return test_dataset
        else:

            train_dataset, val_dataset = train_test_split(list_data, test_size=0.20, random_state=40)
            print("Number of graphs in training dataset = ",len(train_dataset))
            print("Number of graphs in validating dataset = ",len(val_dataset))
            return train_dataset, val_dataset
