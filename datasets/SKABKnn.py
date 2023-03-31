import os
import glob
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from datasets.KNNGraph import KNNGraph
from datasets.AuxFunction import FFT
import pickle
# ------------------------------------------------------------
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
    
    paths = glob.glob("data/SKAB/*/*.csv")
    paths = [p for p in paths if p.find("acc1")>0]

    for j,path in enumerate(paths):
        data1 = data_load(sample_length,filename=path, label=j,InputType=InputType,task=task,overlapping_number = overlapping_number)
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


def data_load(signal_size,filename, label, InputType, task, overlapping_number):
    '''
    This function is mainly used to generate test data and training data.
    filename:Data location
    axisname:Select which channel's data,---->"_DE_time","_FE_time","_BA_time"
    '''
    fl = pd.read_csv(filename, header=None)
    fl = (fl - fl.min()) / (fl.max() - fl.min())
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



class SKABKnn(object):
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
            with open(os.path.join('C:\\Users\\situser\\Desktop\\Vinothini\\SUTD\\PHMGNNBenchmark\\data\\SKAB', "SKABKnn.pkl"), 'wb') as fo:
                print("datadir",self.data_dir)
                pickle.dump(list_data, fo)

        if test:
            test_dataset = list_data
            return test_dataset 
        else:

            train_dataset, val_dataset = train_test_split(list_data, test_size=0.20, random_state=40)

            return train_dataset, val_dataset
