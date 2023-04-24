#!/usr/bin/env python
# coding: utf-8

# In[1]:

import sys
print("Python version")
print (sys.version)

import numpy as np
import io, os, sys
import time

import argparse


#-----------------------argparser sertting---------------------------
parser = argparse.ArgumentParser(description='Exalearn_miniapp_training')

#parser.add_argument('--batch-size', type=int, default=64, metavar='N',
#                    help='input batch size for training (default: 64)')
parser.add_argument('--num_epochs', type=int, default=30, metavar='N',
                    help='number of epochs to train (default: 30)')
parser.add_argument('--device', default='cpu',
                    help='Wheter this is running on cpu or gpu')
parser.add_argument('--phase', type=int, default=0,
                    help='the current phase of workflow, phase0 will not read model')
parser.add_argument('--num_threads', type=int, default=0,
                    help='set number of threads per worker')
parser.add_argument('--data_root_dir', default='./',
                    help='the root dir of gsas output data')
parser.add_argument('--model_dir', default='./',
                    help='the directory where save and load model')
parser.add_argument('--mat_size', type=int, default=3000,
                    help='the matrix with have size of mat_size * mat_size')
#parser.add_argument('--rank_data_gen', type=int, default=256,
#                    help='number of ranks used to generate input data')
args = parser.parse_args()
args.cuda = args.device.find("gpu")!=-1



#-----------------------data loading and spliting---------------------------
root_path = args.data_root_dir + '/phase{}'.format(args.phase) + '/'
print("root_path for data = ", root_path)

msz = args.mat_size
# # load the all_in_one data
print("Start reading and preprocessing data!")

X_scaled = np.load(root_path + 'all_X_data.npy')
y_scaled = np.load(root_path + 'all_Y_data.npy')
print(X_scaled.shape, y_scaled.shape)


X_scaled = np.float32(X_scaled)
y_scaled = np.float32(y_scaled)
X_train, X_test = X_scaled[np.random.randint(msz)], X_scaled[np.random.randint(msz)]
y_train, y_test = y_scaled[np.random.randint(msz)], y_scaled[np.random.randint(msz)]
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
print("X_scale x size:",len(X_scaled))
print("X_scale y size:",len(X_scaled[0]))
print(X_scaled)
print("Y_sclae size:",len(y_scaled))

#TODO We need to consider how we want to do trainin using multiple resources CPU/GPU
t1 = time.time()
for epoch in range(args.num_epochs):
    R=np.matmul(X_scaled, X_scaled)
t2 = time.time()
print ("Time taken to multiply is {} for num of epoch = {}".format(t2-t1, args.num_epochs))

with open(args.model_dir + '/result_phase{}.npy'.format(args.phase), 'wb') as f:
    np.save(f, R)


