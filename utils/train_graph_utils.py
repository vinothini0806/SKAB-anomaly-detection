
# %load utils/train_graph_utils.py
#!/usr/bin/python
# %matplotlib inline
import numpy as np
# from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import logging
import os
import time
import warnings
import torch
from torch import nn
from torch import optim
from torch_geometric.data import DataLoader
import models
import models2
import datasets
from utils.save import Save_Tool
from utils.freeze import set_freeze_by_id
import pandas as pd

def f1_score(y_true, y_pred):
        """
        Calculate the F1 score given the true labels and predicted labels.
        
        Args:
            y_true (array-like): The true labels.
            y_pred (array-like): The predicted labels.
        
        Returns:
            f1_score (float): The F1 score.
        """
        # Calculate the number of true positives, false positives, and false negatives.
        tp = sum((y_true == 0) & (y_pred == 0))
        fp = sum((y_true == 1) & (y_pred == 0))
        fn = sum((y_true == 0) & (y_pred == 1))
              # Calculate precision and recall.
        if tp + fp == 0:
            precision = 0.0
            print("precision",precision)
        else:
            precision = tp / (tp + fp)
            print("precision",precision)

  

        recall = tp / (tp + fn)
        print("recall",recall)
        print("tp",tp)
        print("fn",fn)
        # Calculate the F1 score.
        f1_score = 2 * (precision * recall) / (precision + recall)
        
        return f1_score
class train_utils(object):
    def __init__(self, args, save_dir):
        self.args = args
        self.save_dir = save_dir

    def setup(self):
        """
        Initialize the datasets, model, loss and optimizer
        :param args:
        :return:
        """
        args = self.args

        # Consider the gpu or cpu condition
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            logging.info('using {} gpus'.format(self.device_count))
            assert args.batch_size % self.device_count == 0, "batch size should be divided by device count"
        else:
            warnings.warn("gpu is not available")
            self.device = torch.device("cpu")
            self.device_count = 1
            logging.info('using {} cpu'.format(self.device_count))


        # Load the datasets
        # data name is graph type function for specific dataset which we are going to apply for raw data by using KNN or Radius or Path
        Dataset = getattr(datasets, args.data_name)
        self.datasets = {}
        # sample length -> node feature length
        # data_dir -> the directory of the data as pickle file after apply the KNN or Path or Radius
        # Input_type -> the input type decides the length of input
        # task -> Node classification or Graph classification
        self.datasets['train'], self.datasets['val'] = Dataset(args.sample_length,args.data_dir, args.Input_type, args.task, args.overlapping_number,args.file_name).data_preprare()

        # num_workers = number of training process
        self.dataloaders = {x: DataLoader(self.datasets[x], batch_size=args.batch_size,
                                                           shuffle=(True if x == 'train' else False),
                                                           num_workers=args.num_workers,
                                                           pin_memory=(True if self.device == 'cuda' else False))
                            for x in ['train', 'val']}
        # Define the model
        InputType = args.Input_type
        if InputType == "TD":
            feature = args.sample_length
        elif InputType == "FD":
            feature = int(args.sample_length/2)
        elif InputType == "other":
            feature = 1
        else:
            print("The InputType is wrong!!")

        if args.task == 'Node':
            self.model = getattr(models, args.model_name)(feature=feature,out_channel=Dataset.num_classes)
        elif args.task == 'Graph':
            if args.pretrained_model == 1:
                self.model = getattr(models2, args.model_name)(feature=feature, out_channel=Dataset.num_classes,pooltype=args.pooltype)
                self.model.load_state_dict(torch.load('checkpoint\Graph_GCN_EdgePool_SKABM2Knn_TD_0420-104124\49-0.7899-best_model.pth'))
            else:
                self.model = getattr(models2, args.model_name)(feature=feature, out_channel=Dataset.num_classes,pooltype=args.pooltype)
            
        else:
            print('The task is wrong!')

        if args.layer_num_last != 0:
            # unfrozen the mentioned last layers in the model architecture
            set_freeze_by_id(self.model, args.layer_num_last)
        if self.device_count > 1:
            self.model = torch.nn.DataParallel(self.model)

        # Define the optimizer
        if args.opt == 'sgd':
            self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr,
                                       momentum=args.momentum, weight_decay=args.weight_decay)
        elif args.opt == 'adam':
            self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr,
                                        weight_decay=args.weight_decay)
        else:
            raise Exception("optimizer not implement")

        # Define the learning rate decay
        if args.lr_scheduler == 'step':
            steps = [int(step) for step in args.steps.split(',')]
            self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, steps, gamma=args.gamma)
        elif args.lr_scheduler == 'exp':
            self.lr_scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, args.gamma)
        elif args.lr_scheduler == 'stepLR':
            steps = int(args.steps)
            self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, steps, args.gamma)
        elif args.lr_scheduler == 'fix':
            self.lr_scheduler = None
        else:
            raise Exception("lr schedule not implement")

        # Load the checkpoint
        self.start_epoch = 0
        if args.resume:
            suffix = args.resume.rsplit('.', 1)[-1]
            if suffix == 'tar':
                checkpoint = torch.load(args.resume)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.start_epoch = checkpoint['epoch'] + 1
            elif suffix == 'pth':
                self.model.load_state_dict(torch.load(args.resume, map_location=self.device))

        # Invert the model and define the loss
        self.model.to(self.device)
        self.criterion = torch.nn.BCELoss()
    


    def train(self):
        """
        Training process
        :return:
        """


        args = self.args
        threshold = 0.3
        step = 0
        best_acc = 0.0
        batch_count = 0
        batch_loss = 0.0
        batch_acc = 0
        x = 0
        val_F1_Score = []
        train_F1_Score = []
        step_start = time.time()
        # args.max_model_num -> the number of most recent models to save
        save_list = Save_Tool(max_num=args.max_model_num)
        # max_epoch -> number of epochs
        for epoch in range(self.start_epoch, args.max_epoch):
            
            logging.info('-'*5 + 'Epoch {}/{}'.format(epoch, args.max_epoch - 1) + '-'*5)
            # Update the learning rate
            if self.lr_scheduler is not None:
                logging.info('current lr: {}'.format(self.lr_scheduler.get_last_lr()))
            else:
                logging.info('current lr: {}'.format(args.lr))
            
            # Each epoch has a training and val phase
            for phase in ['train', 'val']:
                # Define the temp variable
                epoch_start = time.time()
                epoch_acc = 0
                epoch_loss = 0.0
                pred_list = []
                label_list = []
                
                tp = 0
                tn = 0
                num_missing_targets = 0
                num_false_targets = 0
                missing_alarm_rate = 0.0
                False_alarm_rate = 0.0
                # Set model to train mode or test mode
                if phase == 'train':
                    # sets the model to train mode,
                    # the model will keep track of the gradients and update its parameters during the training process
                    self.model.train()
                else:
                    # sets the model to evaluation mode,
                    #  which means that the model will not update its parameters and will not keep track of gradients.
                    self.model.eval()
                sample_num = 0
                # for loop for access traing and validation dta during each epoch
                for data in self.dataloaders[phase]:
                    inputs = data.to(self.device)
                    labels = inputs.y
                    
                    x += len(inputs.batch)
                    if args.task == 'Node':
                        bacth_num = inputs.num_nodes
                        sample_num += len(labels)
                    elif args.task == 'Graph':
                        bacth_num = inputs.num_graphs
                        sample_num += len(labels)
                    else:
                        print("There is no such task!!")
                    # Do the learning process, in val, we do not care about the gradient for relaxing
                    with torch.set_grad_enabled(phase == 'train'):

                        # forward
                        if  args.task == 'Node':
                            logits = self.model(inputs)
                            # print("logits",len(logits))
                        elif args.task == 'Graph':
                            logits = self.model(inputs,args.pooltype)
                        else:
                            print("There is no such task!!")
                        labels = torch.unsqueeze(labels, dim=1)
                        labels = labels.float()
                        loss = self.criterion(logits, labels)
                        

                        # pred -> predictions of node labels for univariate data 
                        # pred -> predictions of graph labels for multivariate data 
                        pred = logits
                        if phase == 'val':
                          print("pred val",pred)  
                        pred = (pred > threshold).long()
                        pred_list = pred_list + list(pred)
                        label_list = label_list + list(labels)
                        
                        correct = torch.eq(pred, labels).float().sum().item()
                        loss_temp = loss.item() * bacth_num
                        epoch_loss += loss_temp
                        epoch_acc += correct

                        # Calculate the training information
                        if phase == 'train':
                            # backward
                            self.optimizer.zero_grad()
                            loss.backward()
                            self.optimizer.step()

                            batch_loss += loss_temp
                            batch_acc += correct
                            batch_count += bacth_num

                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

                # Print the train and val information via each epoch

                epoch_loss = epoch_loss / sample_num
                label_list = [tensor.item() for tensor in label_list]
                pred_list = [tensor.item() for tensor in pred_list]
                label_list = np.array(label_list)
                pred_list = np.array(pred_list)
                epoch_acc = f1_score(label_list,pred_list)
                if phase == 'train':
                    train_F1_Score.append(epoch_acc)
                else:
                    val_F1_Score.append(epoch_acc)
                
                for i,label in enumerate(label_list):
                    if label != pred_list[i]:
                        if label==1:
                            num_missing_targets += 1
                        elif label==0:
                            num_false_targets += 1
                    else:
                        if label==1:
                            tn += 1
                        elif label==0:
                            tp += 1
                missing_alarm_rate = num_missing_targets/len(label_list)
                False_alarm_rate = num_false_targets/len(label_list)
                
                logging.info('Epoch: {} {}-Loss: {:.4f} {}-F1Score: {:.4f} {}-missing_rate:{:.4f} {}-false_rate:{:.4f} Cost {:.4f} sec'.format(
                    epoch, phase, epoch_loss, phase, epoch_acc,phase ,missing_alarm_rate, phase,False_alarm_rate, time.time()-epoch_start
                ))

                # save the model
                if phase == 'val':
                    # save the checkpoint for other learning
                    model_state_dic = self.model.module.state_dict() if self.device_count > 1 else self.model.state_dict()
                    save_path = os.path.join(self.save_dir, '{}_ckpt.tar'.format(epoch))
                    torch.save({
                        'epoch': epoch,
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'model_state_dict': model_state_dic
                    }, save_path)
                    save_list.update(save_path)
                    # save the best model according to the val accuracy
                    if epoch_acc > best_acc or epoch > args.max_epoch-2:
                        best_acc = epoch_acc
                        logging.info("save best model epoch {}, acc {:.4f}".format(epoch, epoch_acc))
                        torch.save(model_state_dic,
                                   os.path.join(self.save_dir, '{}-{:.4f}-best_model.pth'.format(epoch, best_acc)))
        print(train_F1_Score)
        print(val_F1_Score)
        plt.plot(train_F1_Score, val_F1_Score)
        plt.show()


        











