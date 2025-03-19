# coding=utf-8
# author: Chenhao Cui
# date:20250221
# email: 3313398924@qq.com
# description: draft for reproducing the results of the paper.
# version: tensorflow.keras without gpu

# library
import os
import pandas as pd
import numpy as np
from typing import List, Union
import pandas as pd
import numpy as np
from scipy.stats import jarque_bera
from statsmodels.stats.diagnostic import acorr_ljungbox
import cbams_bilstm_modules as cbms
from keras.models import Model
from keras.layers import *
from keras.models import load_model

################################################# Multivariate one-step prediction

# load data
base_dir = "E:/cchniklaus/apps/vscode/project/cbams_bilstm"
data_path_list = os.listdir(base_dir + "/01_raw_data")

data_list = []
data_name_list = []
for i in range(len(data_path_list)):
    data_name = data_path_list[i]
    data_name_list.append(data_name.split(sep=".")[0])
    data_df = pd.read_csv(base_dir + "/01_raw_data/" + data_name)
    # we plan to extract 2000 data point for every data file. And the time is up to 2020-08-31.
    # for shanghai.csv and shenzhen.csv, the index is from 5747 to 7746.
    goal_data_df = data_df.iloc[5747:7747,3:]
    goal_data_df.to_csv(base_dir + "/02_goal_data_m/goal_data_%s.csv"% data_name.split(sep=".")[0],index=False)


# preprocess data, then split samples, save them finally.
data_name = ["shanghai","shenzhen"]
for i in range(len(data_name)):
    name = data_name[i]
    all_data_df = pd.read_csv(base_dir + "/02_goal_data_m/goal_data_%s.csv"% name)
    # normalization
    max_df = all_data_df.max(axis=0)
    min_df = all_data_df.min(axis=0)
    norm_all_data_df = (all_data_df - min_df)/(max_df - min_df)
    norm_all_data_array = norm_all_data_df.values
    x, y = cbms.transform_data_m(norm_all_data_array,dim_x=60,dim_y=1)
    x_array = np.array(x)
    y_array = np.array(y)
    x_array_train, x_array_valida, x_array_test = x_array[:-400], x_array[-400:-200], x_array[-200:]
    y_array_train, y_array_valida, y_array_test = y_array[:-400], y_array[-400:-200], y_array[-200:]
    np.save(base_dir + "/03_train_valida_test_m/%s_x_train.npy"% name,x_array_train,allow_pickle=True)
    np.save(base_dir + "/03_train_valida_test_m/%s_x_valida.npy"% name,x_array_valida,allow_pickle=True)
    np.save(base_dir + "/03_train_valida_test_m/%s_x_test.npy"% name,x_array_test,allow_pickle=True)
    np.save(base_dir + "/03_train_valida_test_m/%s_y_train.npy"% name,y_array_train,allow_pickle=True)
    np.save(base_dir + "/03_train_valida_test_m/%s_y_valida.npy"% name,y_array_valida,allow_pickle=True)
    np.save(base_dir + "/03_train_valida_test_m/%s_y_test.npy"% name,y_array_test,allow_pickle=True)

# train model
data_name = ["shanghai","shenzhen"]
for i in range(len(data_name)):
    name = data_name[i]
    x_array_train = np.load(base_dir + "/03_train_valida_test_m/%s_x_train.npy"% name)
    x_array_valida = np.load(base_dir + "/03_train_valida_test_m/%s_x_valida.npy"% name)
    y_array_train = np.load(base_dir + "/03_train_valida_test_m/%s_y_train.npy"% name)
    y_array_valida = np.load(base_dir + "/03_train_valida_test_m/%s_y_valida.npy"% name)

    len_t = x_array_train.shape[1]
    num_feats = x_array_train.shape[2]

    input_layer = Input(batch_shape=(None,len_t,num_feats))
    model_list = [cbms.cnn,cbms.lstm,cbms.bilstm,cbms.cnn_lstm,cbms.cnn_bilstm]
    output_layer_list = [model_list[i](input_layer) for i in range(len(model_list))]
    output_layer_list.append(cbms.cbams_bilstm(input_layer,cbam_n=15))
    model_name = ["cnn_m","lstm_m","bilstm_m","cnn_lstm_m","cnn_bilstm_m","cbmams_bilstm_m"]

    for i in range(len(output_layer_list)):
        cbms.train_model(input_layer=input_layer,
                         output_layer=output_layer_list[i],
                         x_y_train=(x_array_train,y_array_train),
                         x_y_valida=(x_array_valida,y_array_valida),
                         loss_path=base_dir + "/04_loss_m/%s_%s_mse.png"% (name,model_name[i]),
                         model_path=base_dir + "/05_model_m/%s_%s.h5"% (name,model_name[i]))

# evaluate model
mcs = cbms.Mcstest()
mcs_results = []
data_name = ["shanghai","shenzhen"]
for i in range(len(data_name)):
    print(i)
    name = data_name[i]
    x_array_test = np.load(base_dir + "/03_train_valida_test_m/%s_x_test.npy"% name)
    y_array_test = np.load(base_dir + "/03_train_valida_test_m/%s_y_test.npy"% name)
    len_t = x_array_test.shape[1]
    num_feats = x_array_test.shape[2]
    # load model
    model_name = ["cnn_m","lstm_m","bilstm_m","cnn_lstm_m","cnn_bilstm_m","cbmams_bilstm_m"]
    model_list = []
    pred_y_dict = {}
    metrics_results_list = []

    for j in range(len(model_name)):
        model = load_model(base_dir + "/05_model_m/%s_%s.h5"% (name,model_name[j]))
        # predict
        pred_y = model.predict(x_array_test.reshape((-1,len_t,num_feats)))
        pred_y_dict[model_name[j]] = np.squeeze(pred_y)
        # evaluate
        metrics_results,metrics_name = cbms.evaluate_model(y_array_test.flatten(),pred_y.flatten())
        metrics_results_list.append([model_name[j]]+metrics_results)
    #save metrics
    results_df = pd.DataFrame(metrics_results_list,columns=["model"]+metrics_name)
    results_df.to_csv(base_dir + "/06_results_m/%s_metrics.csv"% name,index=False)
    # compute mcs test
    error_list,error_name = mcs.error(real_y=y_array_test.flatten(),pred_y=pred_y_dict)
    mcs_results.append([i.reshape((-1,i.shape[0],i.shape[1])) for i in error_list])

# compute mean
# transform dataframe to array
error_name = ["mse","mae","hmse","hmae"]
mean_mcs_results = []
for i in range(len(error_name)):
    s = np.concatenate([j[i] for j in mcs_results],axis=0)
    mean_mcs_results.append(np.mean(s,axis=0))
# save
mcs_dict = {}
for stats in ["R","MAX"]:
    for j in range(len(error_name)):
        error = mean_mcs_results[j]
        name = error_name[j] + "_" + stats
        mcs_dict[name] = np.squeeze(mcs.mcs_test(error,stats))
mcs_df = pd.DataFrame(mcs_dict,index=["cnn_m","lstm_m","bilstm_m","cnn_lstm_m","cnn_bilstm_m","cbmams_bilstm_m"])
mcs_df.to_csv(base_dir + "/06_results_m/mcs.csv",index=True)
