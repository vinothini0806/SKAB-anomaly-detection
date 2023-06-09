
from datasets.Generator import Gen_graph


def PathGraph(interval,data,label,task):
    a, b = 0, interval
    graph_list = []
    if task == 'Node':
        while b <= len(data):
            graph_list.append(data[a:b])
            a += interval
            b += interval
    else:
        graph_list = data
    graphset = Gen_graph("PathGraph",graph_list,label,task)
    return graphset