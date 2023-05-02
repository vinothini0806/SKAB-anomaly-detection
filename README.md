# Graph Convolutional Neural Netwrok
## [The emerging graph convolutional neural network for anomaly detection in SKAB Dataset](https://www.sciencedirect.com/science/article/pii/S0888327021009791)



# Implementation of the paper:
Paper:
```
@article{PHMGNNBenchmark,
  title={The emerging graph neural networks for intelligent fault diagnostics and prognostics: A guideline and a benchmark study},
  author = {Tianfu Li and Zheng Zhou and Sinan Li and Chuang Sun and Ruqiang Yan and Xuefeng Chen},
  journal={Mechanical Systems and Signal Processing},
  volume = {168},
  pages = {108653},
  year = {2022},
  issn = {0888-3270},
  doi = {https://doi.org/10.1016/j.ymssp.2021.108653},
  url = {https://www.sciencedirect.com/science/article/pii/S0888327021009791},
}
```

![PHMGNNBenchmark](https://github.com/HazeDT/PHMGNNBenchmark/blob/main/Framework.png)

# Requirements
* Python 3.8 or newer
* torch-geometric 1.6.1
* pytorch  1.6.0
* pandas  1.0.5
* numpy  1.18.5

# Guide 
 We provide anomaly detection framework based on GCN ( Graph Convolutional Neural Network. The framework consists of two branches, that is, the node-level fault diagnostics architecture and graph-level fault diagnostics. In node-level fault diagnosis, each node of a graph is considered as a sample, while the entire graph is considered as a sample in graph-level fault diagnosis. <br> In this code library, we provide three graph constrcution methods (`KnnGraph`, `RadiusGraph`, and `PathGraph`), and two different input types (`Frequency domain` and `time domain`). Besides, seven GNNs and four graph pooling methods are implemented. 
 
# Pakages
* `datasets` contians the data load method for different dataset
* `model` contians the implemented GCN model for nodel-level task
* `model2` contians the implemented GCN model for graph-level rask

# Run the code
  * Node level anomaly detection <br>
  python  ./train_graph_diagnosis.py --model_name GCN --data_name SKABMKnn --data_dir ./data/SKABM/SKABMKnn.pkl  --Input_type TD  --task Node   --checkpoint_dir ./checkpoint --file_name df_valve1
  * Graph level anomaly detection <br>
  python  ./train_graph_diagnosis.py --model_name GCN --data_name SKABMKnn --data_dir ./data/SKABM --Input_type TD  --task Graph --pooltype EdgePool  --checkpoint_dir ./checkpoint --file_name df_valve1
  
## The data for runing the demo
   In order to facilitate your implementation, we give some organized data here for node level-fault diagnosis and graph-level prognosis [`Data for demo`]
   * `SKABAcc1` contians the Accelerometer1 values within different folders ( eg:- df_valve1, df_valve2). Each folder indicates different conditions as shown in this [data structure](https://github.com/vinothini0806/SKAB-anomaly-detection/blob/main/folder_structure.JPG)
   * As SKABAcc1 folder `SKABAcc2` & `SKABAttr8` folders also contains Accelerometer 2 values and 8 attributes values respectively.
   
# Datasets
## Anomaly detection dataset
* [SKAB Dataset](https://www.kaggle.com/datasets/yuriykatser/skoltech-anomaly-benchmark-skab)



# Note
This code library is run under the `windows operating system`. If you run under the `linux operating system`, you need to delete the `‘/tmp’` before the path in the `dataset` to avoid path errors.

