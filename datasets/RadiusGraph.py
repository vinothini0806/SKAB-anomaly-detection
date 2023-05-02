import torch
from math import sqrt
import numpy as np
from datasets.Generator import Gen_graph

# interval means the number nodes in the each graph
def RadiusGraph(interval,data,label,task):
    a, b = 0, interval
    graph_list = []
    if task == 'Node':
        while b <= len(data):
            graph_list.append(data[a:b])
            a += interval
            b += interval
    else:
        graph_list = data
    graphset = Gen_graph("RadiusGraph",graph_list,label,task)
    return graphset