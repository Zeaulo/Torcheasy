#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Project: Torcheasy: Training models without pain
# Author: Zeaulo
# Github: Zeaulo
# Email : psymhmch@outlook.com
# Created time: 2023-6-26 17:20

from framework import torcheasy

Easy                    = torcheasy()
Easy.gpu_num            = 1
Easy.classes            = 10
Easy.seed               = 34
Easy.train_data_path    = 'dataset/train_data.npy'
Easy.train_label_path   = 'dataset/train_label.npy'
Easy.val_data_path      = 'dataset/val_data.npy'
Easy.val_label_path     = 'dataset/val_label.npy'
Easy.model_code_dir     = 'models'
Easy.model_save_dir     = 'checkpoints'
Easy.logs_save_dir      = 'logs'
Easy.result_save_dir    = 'results'
# Training information
Easy.batch_size         = 8
Easy.set_epoch          = 100
Easy.early_stop         = 10
Easy.learning_rate      = 5e-6
Easy.weight_decay       = 1e-2
Easy.optimizer          = 'NAdam'
# Model information
Easy.consideration_num  = 3
Easy.model_name         = 'model_ResnetLSTM_model'

Easy.train()