#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Project      : Torcheasy: Training models without pain
# Author       : Zeaulo
# Github       : Zeaulo
# Email        : psymhmch@outlook.com
# Created time : 2023-6-26 17:20

# 2.Torcheasy framework - https://github/zeaulo/torcheasy

import torch
import numpy as np
import importlib
import time
from sklearn import metrics
import matplotlib.pyplot as plt
import sys
import os

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

class torcheasy:
    # Preparation in advance
    instruction = ''' -- Changing Parameters --
Quickly COPY:
# Basic information
Easy                    = torcheasy()
Easy.gpu_num            = 1
Easy.classes            = 2
Easy.seed               = 34
Easy.train_data_path    = 'data/train_data.npy'
Easy.train_label_path   = 'data/train_label.npy'
Easy.val_data_path      = 'data/val_data.npy'
Easy.val_label_path     = 'data/val_label.npy'
Easy.model_code_dir     = 'models'
Easy.model_save_dir     = 'checkpoints'
Easy.logs_save_dir      = 'logs'
Easy.result_save_dir    = 'results'
# Training information
Easy.batch_size         = 64
Easy.set_epoch          = 100
Easy.early_stop         = 10
Easy.learning_rate      = 1e-3
Easy.weight_decay       = 1e-4
Easy.optimizer          = 'NAdam'
# Model information
Easy.consideration_num  = 3
Easy.model_name         = 'test_model'

Basic information:
    gpu_num            - (format:int) How many GPUs that you want to use.
    classes            - (format:int) Number of classes.
    seed               - (format:int) Fixed seed for reproducibility of results.
    train_data_path    - (format:str) Path of train data.
    train_label_path   - (format:str) Path of train label.
    val_data_path      - (format:str) Path of validation data.
    val_label_path     - (format:str) Path of validation label.
    model_save_dir     - (format:str) Where to save the model.
    logs_save_dir      - (format:str) Where to save the logs.
    result_save_dir    - (format:str) Where to save the results.

Training information:
    batch_size         - (format:int) Batch size.
    set_epoch          - (format:int) How many epochs to train.
    early_stop         - (format:int) About training process.
    learning_rate      - (format:float) Learning rate.
    weight_decay       - (format:float) L2 regularization.
    optimizer          - (format:str) chose one: ['NAdam', 'Adam', 'SparseAdam', 'RAdam', 'AdamW', 'Adamax', 'Adadelta', 'Adagrad', 'ASGD', 'SGD', 'RMSprop', 'LBFGS', 'Rprop', 'RMSprop']

Model information:
    consideration_num  - (format:int, >=1) Consideration number of compared model in models list.
    model_name         - (format:str) Model name for saving model's name and loading model's structure.

    '''

    def __init__(self):
        # Basic information
        self.device             = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gpu_num            = 1 # Needing look! How many GPUs that you want to use
        self.classes            = 2 # Number of classes
        self.seed               = 34 # Fixed seed for reproducibility of results
        self.train_data_path    = 'data/train_data.npy'
        self.train_label_path   = 'data/train_label.npy'
        self.val_data_path      = 'data/val_data.npy'
        self.val_label_path     = 'data/val_label.npy'
        self.model_code_dir     = 'models'
        self.model_save_dir     = 'checkpoints'
        self.logs_save_dir      = 'logs'
        self.result_save_dir    = 'results'
        if os.path.exists(self.model_save_dir)   == False:
            os.makedirs(self.model_save_dir)
        if os.path.exists(self.logs_save_dir)    == False:
            os.makedirs(self.logs_save_dir)
        if os.path.exists(self.result_save_dir)  == False:
            os.makedirs(self.result_save_dir)
        
        # Training information
        self.batch_size         = 64 
        self.set_epoch          = 100
        self.early_stop         = 10 # About training process
        self.learning_rate      = 1e-3
        self.weight_decay       = 1e-4 # L2 regularization
        self.optimizer          = 'NAdam' # ['NAdam', 'Adam', 'SparseAdam', 'RAdam', 'AdamW', 'Adamax', 'Adadelta', 'Adagrad', 'ASGD', 'SGD', 'RMSprop', 'LBFGS', 'Rprop', 'RMSprop']
        if self.optimizer not in ['NAdam', 'Adam', 'SparseAdam', 'RAdam', 'AdamW', 'Adamax', 'Adadelta', 'Adagrad', 'ASGD', 'SGD', 'RMSprop', 'LBFGS', 'Rprop', 'RMSprop']:
            raise ValueError('Optimizer is not in the list! Please check it!')

        
        # Model information
        self.consideration_num  = 3 # Consideration number of compared model in models list
        self.model_name         = 'test_model' # Model name

        torch.seed = self.seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
    
    # Information recording
    def show_information(self):
        print(f'*'*50)
        print('- - Model Instructure - -')
    
    # Training
    def train(self):
        # Clear the screen
        print('\033c', end='')
        # printing information
        # config information
        print(f'{"*"*50}\n{"* "*5} Config Information {"* "*5}')


        sys.stdout = Logger(f'{self.logs_save_dir}/{self.model_name}.log')
        # Data loading
        train_data         = torch.from_numpy(np.load(self.train_data_path, allow_pickle=True)).float()
        train_label        = torch.from_numpy(np.load(self.train_label_path, allow_pickle=True)).long()
        val_data           = torch.from_numpy(np.load(self.val_data_path, allow_pickle=True)).float()
        val_label          = torch.from_numpy(np.load(self.val_label_path, allow_pickle=True)).long()
        train_dataset      = torch.utils.data.TensorDataset(train_data, train_label)
        val_dataset        = torch.utils.data.TensorDataset(val_data, val_label)
        self.train_loader  = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False)
        self.val_loader    = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        # Model loading
        self.model         = importlib.import_module(self.model_code_dir + '.' + self.model_name).model().to(self.device)
        if self.device     != 'cpu' and self.gpu_num > 1:
            self.model     = torch.nn.DataParallel(self.model, device_ids=list(range(self.gpu_num)))
        
        # Optimizer loading
        if self.device     != 'cpu' and self.gpu_num > 1:
            exec(f"self.optimizer = torch.optim.{self.optimizer}(self.model.module.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)")
            self.optimizer = torch.nn.DataParallel(self.optimizer, device_ids=list(range(self.gpu_num)))
        else:
            exec(f"self.optimizer = torch.optim.{self.optimizer}(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)")
        
        # Loss function loading
        self.loss_func     = torch.nn.CrossEntropyLoss()

        best_epoch         = 0
        best_val_loss      = 1000000
        best_val_acc       = 0
        best_train_loss    = 100000
        train_acc_list     = []
        train_loss_list    = []
        val_acc_list       = []
        val_loss_list      = []
        best_model_loss    = 100000
        models_list        = [i for i in range(self.consideration_num)]
        start_time         = time.time()

        for epoch in range(self.set_epoch):
            self.model.train()
            train_loss = 0
            train_acc = 0
            for i, (x, y) in enumerate(self.train_loader):
                x          = x.to(self.device)
                y          = y.to(self.device)
                outputs    = self.model(x)
                
                loss       = self.loss_func(outputs, y)
                train_loss += loss.item()
                self.optimizer.zero_grad()
                loss.backward()

                if self.device != 'cpu' and self.gpu_num > 1:
                    self.optimizer.module.step()
                else:
                    self.optimizer.step()
                
                _, predicted = torch.max(outputs.data, 1)

                train_acc += (predicted == y).sum().item()
            
            if self.classes > 2:
                average_mode = 'macro'
            else:
                average_mode = 'binary'
            
            train_f1 = metrics.f1_score(y.cpu(), predicted.cpu(), average=average_mode)
            train_pre = metrics.precision_score(y.cpu(), predicted.cpu(), average=average_mode)
            train_recall = metrics.recall_score(y.cpu(), predicted.cpu(), average=average_mode)


            train_loss /= len(self.train_loader)
            train_acc /= len(self.train_loader.dataset)
            train_acc_list.append(train_acc)
            train_loss_list.append(train_loss)
            print('-'*50)
            print('Epoch [{}/{}]\n Train Loss: {:.4f}, Train Acc: {:.4f}'.format(epoch + 1, self.set_epoch, train_loss, train_acc))
            print('Train-f1: {:.4f}, Train-precision: {:.4f} Train-recall: {:.4f}'.format(train_f1, train_pre, train_recall))

            # Validation
            self.model.eval()
            val_loss = 0
            val_acc = 0

            for j, (x, y) in enumerate(self.val_loader):
                x = x.to(self.device)
                y = y.to(self.device)
                outputs = self.model(x)
                loss = self.loss_func(outputs, y)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_acc += (predicted == y).sum().item()

            val_f1 = metrics.f1_score(y.cpu(), predicted.cpu(), average=average_mode)
            val_pre = metrics.precision_score(y.cpu(), predicted.cpu(), average=average_mode)
            val_recall = metrics.recall_score(y.cpu(), predicted.cpu(), average=average_mode)

            val_loss /= len(self.val_loader)
            val_acc /= len(self.val_loader.dataset)
            val_acc_list.append(val_acc)
            val_loss_list.append(val_loss)
            print('\nVal Loss: {:.4f}, Val Acc: {:.4f}'.format(val_loss, val_acc))
            print('Val-f1: {:.4f}, Val-precision: {:.4f} Val-recall: {:.4f}'.format(val_f1, val_pre, val_recall))
            print(f'Best val acc is {best_val_acc:.4f}.')
            distance = self.early_stop - (epoch+1 - best_epoch)
            print(f'\nTip: Best train loss is {best_train_loss:.4f}, and best val loss is {best_val_loss:.4f}.\nThe distance of stopping is {distance} now :)')

            # Choose the best model for saving1
            # (epoch+1)%save_num -> Replace the old model in the list
            models_list[(epoch+1)%self.consideration_num] = self.model
            if epoch+1 >= self.consideration_num:
                # -save_num:-1 -> The model score list is always in the last few numbers
                models_list_loss = val_loss_list[-self.consideration_num:-1]
                # model_loss -> model_loss is the average loss of the model list
                # The purpose is to find the most stable model list
                model_loss = np.mean(models_list_loss)
                if model_loss <= best_model_loss:
                    best_model_loss = model_loss
                    # Choose the best model in the model list -> The best model in the most stable model list
                    perfect = np.argmin(models_list_loss)
                    print(f'-> best_model_loss {model_loss} has accessed, saving model...')
                    if self.device == 'cuda' and self.gpu_num > 1:
                        torch.save(models_list[perfect].module.state_dict(), f'{self.model_save_dir}/{self.model_name}.pth')
                    else:
                        torch.save(models_list[perfect].state_dict(), f'{self.model_save_dir}/{self.model_name}.pth')
            if best_val_acc < val_acc:
                best_val_acc = val_acc
                best_val_loss = val_loss
            
            if train_loss < best_train_loss:
                best_train_loss = train_loss
                best_val_loss = val_loss
                best_epoch = epoch + 1
            
            # Early stopping
            if epoch+1 - best_epoch == self.early_stop:
                print(f'{self.early_stop} epochs later, the loss of the validation set no longer continues to decrease, so the training is stopped early.')
                end_time = time.time()
                print(f'Total time is {end_time - start_time}s.')
                break

            # Draw the accuracy and loss function curves of the training set and the validation set
            plt.figure()
            plt.plot(range(1, len(train_acc_list) + 1), train_acc_list, label='train_acc')
            plt.plot(range(1, len(val_acc_list) + 1), val_acc_list, label='val_acc')
            plt.legend()
            plt.savefig(f'{self.result_save_dir}/{self.model_name}_acc.png')
            plt.close()
            plt.figure()
            plt.plot(range(1, len(train_loss_list) + 1), train_loss_list, label='train_loss')
            plt.plot(range(1, len(val_loss_list) + 1), val_loss_list, label='val_loss')
            plt.legend()
            plt.savefig(f'{self.result_save_dir}/{self.model_name}_loss.png')
            plt.close()

