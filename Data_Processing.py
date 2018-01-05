import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd


def load_data(parent_dir, file_title, train_folds, dev_folds, test_folds):
    train_set = []
    dev_set = []
    test_set = []
    
    for i in train_folds:
        ds_filename = parent_dir + file_title + str(i)+".csv"
        df = pd.read_csv(ds_filename, index_col = None)
        train_set.append(df)
        
    for i in dev_folds:
        ds_filename = parent_dir + file_title + str(i)+".csv"
        df = pd.read_csv(ds_filename, index_col = None)
        dev_set.append(df)
        
    for i in test_folds:
        ds_filename = parent_dir + file_title + str(i)+".csv"
        df = pd.read_csv(ds_filename, index_col = None)
        test_set.append(df)
        
    print("done!")    
    return  pd.concat(train_set, ignore_index=True), pd.concat(dev_set, ignore_index=True), pd.concat(test_set, ignore_index=True)  


def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels,n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode
