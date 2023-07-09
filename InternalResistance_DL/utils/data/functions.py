import numpy as np
import pandas as pd
import torch
import time

from sklearn import preprocessing
import torch

def load_features(feat_path, dtype=np.float32):
    #feat_df = pd.read_csv(feat_path)
    #feat = np.array(feat_df, dtype=dtype)
    feat = np.load(feat_path)['VandI']
    label=np.load(feat_path)['label']
    ID=np.load(feat_path)['ID']
    ID_code=[]
    for i in range(len(ID)):
        code_temp=[ord(ID[i][j]) for j in range(0,5)]
        #print([ord(ID[i][j]) for j in range(0,5)])

        code_temp=[int(ID[i][j]) for j in range(5,11)]
        ID_code.append(code_temp)
    ID_code=np.array(ID_code)
    CandT=np.load(feat_path)['CandT']
    print(feat.shape,label.shape,ID_code.shape,CandT.shape,'shape')

    return feat, label,ID_code,CandT


def generate_dataset(
    data,data2, ID,CandT,datav,datav2, IDv,CandTv,seq_len, pre_len, time_len=None, split_ratio=0.8, normalize=True
):
    """
    :param data: feature matrix
    :param seq_len: length of the train data sequence
    :param pre_len: length of the prediction data sequence
    :param time_len: length of the time series in total
    :param split_ratio: proportion of the training set
    :param normalize: scale the data to (0, 1], divide by the maximum value in the data
    :return: train set (X, Y) and test set (X, Y)
    """
    if time_len is None:
        time_len = data.shape[0]
    if normalize:
        max_val = np.max(data)
        max_val_label=np.max(data2)
        data = data / max_val
        data2=data2/max_val_label
        max_valv = np.max(datav)
        max_val_labelv = np.max(datav2)
        datav = datav / max_valv
        datav2 = datav2 / max_val_labelv
    train_size = int(time_len * split_ratio)
    train_data = data[:train_size]
    test_data = datav[:train_size]#data[train_size:time_len]
    train_data2 = data2[:train_size]#label
    test_data2 = datav2[:train_size]#data2[train_size:time_len]
    train_ID=ID[:train_size]
    val_ID=IDv[:train_size]#ID[train_size:time_len]
    train_T=CandT[:train_size]
    val_T=CandTv[:train_size]#CandT[train_size:time_len]
    train_X, train_Y, val_X, val_Y,test_X,test_Y = train_data, train_data2, test_data, test_data2,data,data2

    return np.array(train_X).astype('float64'), np.array(train_Y).astype('float64'), np.array(val_X).astype('float64'), np.array(val_Y).astype('float64'),np.array(test_X).astype('float64'), np.array(test_Y).astype('float64'),train_ID,val_ID,ID,train_T,val_T,CandT


def generate_torch_datasets(
    data,data2,ID,CandT,datav,datav2,IDv,CandTv, seq_len, pre_len, time_len=None, split_ratio=0.8, normalize=True
):
    train_X, train_Y,val_X,val_Y, test_X, test_Y ,train_ID,val_ID,test_ID,train_T,val_T,test_T= generate_dataset(
        data,
        data2,
        ID,
        CandT,
        datav,
        datav2,
        IDv,
        CandTv,
        seq_len,
        pre_len,
        time_len=time_len,
        split_ratio=split_ratio,
        normalize=normalize,
    )
    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(train_X), torch.FloatTensor(train_Y),torch.Tensor(train_ID),torch.FloatTensor(train_T)
    )
    val_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(val_X), torch.FloatTensor(val_Y), torch.Tensor(val_ID), torch.FloatTensor(val_T)
    )
    test_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(test_X), torch.FloatTensor(test_Y),torch.Tensor(test_ID),torch.FloatTensor(test_T)
    )
    return train_dataset,val_dataset, test_dataset
