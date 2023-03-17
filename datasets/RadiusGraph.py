import torch
from math import sqrt
import numpy as np
from datasets.Generator import Gen_graph

# interval means the number nodes in the each graph
def RadiusGraph(interval,data,label,task):
    a, b = 0, interval
    graph_list = data
    # print()
    # len(data) = > Number of total nodes
    # while b <= len(data):
    #     # graph_list => contains the graphs which includes nodes
    #     graph_list.append(data[a:b])
    #     a += interval
    #     b += interval
    graphset = Gen_graph("RadiusGraph",graph_list,label,task)
    # print("Number of graphs",len(graphset))
    return graphset