import torch
from math import sqrt
import numpy as np
from torch_geometric.data import Data
from scipy.spatial.distance import pdist
import copy

def KNN_classify(k,X_set,x):
    """
    k:number of neighbours
    X_set: the datset of x
    x: to find the nearest neighbor of data x
    """

    distances = [sqrt(np.sum((x_compare-x)**2)) for x_compare in X_set]
    nearest = np.argsort(distances)
    node_index  = [i for i in nearest[1:k+1]]
    topK_x = [X_set[i] for i in nearest[1:k+1]]
    # print("node_index",node_index)
    return  node_index,topK_x


def KNN_weigt(x,topK_x):
    distance = []
    v_1 = x
    data_2 = topK_x
    for i in range(len(data_2)):
        v_2 = data_2[i]
        
        combine = np.vstack([v_1, v_2])
        likely = pdist(combine, 'euclidean')
        distance.append(likely[0])
    beata = np.mean(distance)
    w = np.exp((-(np.array(distance)) ** 2) / (2 * (beata ** 2)))
    return w


def KNN_attr(data):
    '''
    for KNNgraph
    :param data:
    :return:
    '''
    edge_raw0 = []
    edge_raw1 = []
    edge_fea = []
    for i in range(len(data)):
        # x ->single node in each 
        # data-> each graph
        x = data[i]
        # print("len(data)",len(data))
        if len(data) == 2:
           node_index, topK_x= KNN_classify(1,data,x)
           local_index = np.zeros(1)+i
        elif len(data) == 3:
            node_index, topK_x= KNN_classify(2,data,x)
            local_index = np.zeros(2)+i
        elif len(data) > 3 :
            node_index, topK_x= KNN_classify(5,data,x)
            local_index = np.zeros(5)+i
        else:
            print("Invalid input graph")
        
        loal_weigt = KNN_weigt(x,topK_x)
        

        edge_raw0 = np.hstack((edge_raw0,local_index))
        edge_raw1 = np.hstack((edge_raw1,node_index))
        edge_fea = np.hstack((edge_fea,loal_weigt))

    edge_index = [edge_raw0, edge_raw1]
    # print("edge_raw0",edge_raw0)
    # print("edge_raw1",edge_raw1)
    return edge_index, edge_fea


# data - > graph with list of nodes, S1, S2 - >  nodes which contains the node features
def cal_sim(data,s1,s2):
    edge_index = [[],[]]
    edge_feature = []
    # If condition to consider the nodes with other nodes except the node itself
    if s1 != s2:
        # v_1,v_2  -> different nodes in the graph where each nodes sontains the same number of features
        v_1 = data[s1]
        v_2 = data[s2]
        # verticaly stacked the two lists such as v_1 and v_2
        combine = np.vstack([v_1, v_2])
        # pdist() returns a condensed distance matrix,
        #  which is just a 1D array that contains the pairwise distances between the vectors.
        # Since we want to compute cosine similarity instead of cosine distance,
        #  we subtract the distance from 1 to get the similarity values. 
        # likely array will have n elements, where
        #  each element represents the cosine similarity between the corresponding pairs of vectors from v_1 and v_2
        likely = 1- pdist(combine, 'cosine')
#         w = np.exp((-(likely[0]) ** 2) / 30)
        # item() method is called on the resulting NumPy array to extract the scalar value of the similarity score.
        if likely.item() >= 0:
            # W -> weight of edge between above two vertices v_1 and v_2
            w = 1
            edge_index[0].append(s1)
            edge_index[1].append(s2)
            edge_feature.append(w)
            # edge_index is 2 d array which contains pair of vertices which contains connections
    return edge_index,edge_feature


# here in data is a graph with list of nodes
def Radius_attr(data):
    '''
    for RadiusGraph
    :param feature:
    :return:
    '''
    #  range(len(data)) -> creates a sequence of numbers from 0 to len(data) - 1/ from 0 to (number of nodes in graph -1) 
    s1 = range(len(data))
    # creates a new copy of s1 such that any changes made to s2 do not affect s1.
    s2 = copy.deepcopy(s1)
    # np.array([[], []]) is a way of creating a 2D NumPy array with two empty sub-arrays
    edge_index = np.array([[], []])  # 一个故障样本与其他故障样本匹配生成一次图
    edge_fe = []
    for i in s1:
        for j in s2:
            # cal_sim ->  returns the edge between the nodes and the weight
            local_edge, w = cal_sim(data, i, j)
            # edge_index contains the nodes which have edges
            edge_index = np.hstack((edge_index, local_edge))
            if any(w):
                edge_fe.append(w[0])
            # edge_fe contains the edgeweights for the relevant edges
    return edge_index,edge_fe


def Path_attr(data):

    node_edge = [[], []]

    for i in range(len(data) - 1):
        node_edge[0].append(i)
        node_edge[1].append(i + 1)

    distance = []
    for j in range(len(data) - 1):
        v_1 = data[j]
        v_2 = data[j + 1]
        combine = np.vstack([v_1, v_2])
        likely = pdist(combine, 'euclidean')
        distance.append(likely[0])

    beata = np.mean(distance)
    w = np.exp((-(np.array(distance)) ** 2) / (2 * (beata ** 2)))  #Gussion kernel高斯核

    return node_edge, w

# Here data means the graph list
def Gen_graph(graphType, data, label,task):
    data_list = []
    # edge_index = []
    if graphType == 'KNNGraph':
        for i in range(len(data)):
            graph_feature = data[i]
            if task == 'Node':
                labels = np.zeros(len(graph_feature)) + label
            elif task == 'Graph':
                labels = [label]
            else:
                print("There is no such task!!")
            node_edge, w = KNN_attr(data[i])
            node_features = torch.tensor(graph_feature, dtype=torch.float)
            graph_label = torch.tensor(labels, dtype=torch.long)  # 获得图标签
            # print("length of node_edge",len(node_edge[0][0]))
            # for i,value in enumerate(node_edge):
            #     edge_index[i] = np.array(value)
            # edge_index = np.concatenate(node_edge, axis=0)
            # print("len(node_edge)",len(node_edge[1][0]))
            node_edge = np.array(node_edge)
            # print("node_edge_array",node_edge)
            # print("len(node_edge)",len(node_edge))
            # node_edge = node_edge.astype('float')
            edge_index = torch.tensor(node_edge, dtype=torch.long)
            edge_features = torch.tensor(w, dtype=torch.float)
            # Generate graphs using 
            graph = Data(x=node_features, y=graph_label, edge_index=edge_index, edge_attr=edge_features)
            data_list.append(graph)

    elif graphType == 'RadiusGraph':
        # len(data) -> number of graphs
        for i in range(len(data)):
            # graph_feature - > each graph
            graph_feature = data[i]
            if task == 'Node':
                # labels -> add label for each node in the graph
                labels = np.zeros(len(graph_feature)) + label
            elif task == 'Graph':
                # labels -> it is a list with single value as label
                labels = [label]
            else:
                print("There is no such task!!")
            # node_edge -> edges in the graph, w -> of each edges in the graph
            node_edge, w = Radius_attr(graph_feature)
            # convert graph(node_features), graph_label,edge_index and  edge_features as tensors
            node_features = torch.tensor(graph_feature, dtype=torch.float)
            graph_label = torch.tensor(labels, dtype=torch.long)  # 获得图标签
            edge_index = torch.tensor(node_edge, dtype=torch.long)
            # edge_features -> weights of each edge in the graph
            edge_features = torch.tensor(w, dtype=torch.float)
            # generate graph using torch_geometric library
            graph = Data(x=node_features, y=graph_label, edge_index=edge_index, edge_attr=edge_features)
            # data_list -> append each graph to list called data_list
            data_list.append(graph)

    elif graphType == 'PathGraph':
        for i in range(len(data)):
            graph_feature = data[i]
            if task == 'Node':
                labels = np.zeros(len(graph_feature)) + label
            elif task == 'Graph':
                labels = [label]
            else:
                print("There is no such task!!")
            node_edge, w = Path_attr(graph_feature)
            node_features = torch.tensor(graph_feature, dtype=torch.float)
            graph_label = torch.tensor(labels, dtype=torch.long)  # 获得图标签
            edge_index = torch.tensor(node_edge, dtype=torch.long)
            edge_features = torch.tensor(w, dtype=torch.float)
            graph = Data(x=node_features, y=graph_label, edge_index=edge_index, edge_attr=edge_features)
            data_list.append(graph)

    else:
        print("This GraphType is not included!")
    return data_list
