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


################################################## Autoregressive multistep prediction

# load data and compute statistics
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
    goal_data_df = data_df.iloc[5747:7747,6]
    data_list.append(goal_data_df)

# concatenate data
all_data_df = pd.concat(data_list,axis=1,ignore_index=True,join="outer")
all_data_df.columns = data_name_list
all_data_df.to_csv(base_dir + "/02_goal_data/goal_data.csv",index=False)

# get dataset

all_data_df = pd.read_csv(base_dir + "/02_goal_data/goal_data.csv")
# normalization
max_df = all_data_df.max(axis=0)
min_df = all_data_df.min(axis=0)
norm_all_data_df = (all_data_df - min_df)/(max_df - min_df)
# transformation
data_name = list(norm_all_data_df.columns)
norm_data_list = norm_all_data_df.values.T.tolist()
transform_data_list = [cbms.transform_data(i,dim_x=60,dim_y=8) for i in norm_data_list]
# save
for i in range(len(transform_data_list)):
    x_y = transform_data_list[i]
    name = data_name[i]
    x_y_train, x_y_valida, x_y_test = x_y[:-400], x_y[-400:-200], x_y[-200:]
    x_y_train_df = pd.DataFrame(x_y_train)
    x_y_valida_df = pd.DataFrame(x_y_valida)
    x_y_test_df = pd.DataFrame(x_y_test)
    x_y_train_df.to_csv(base_dir + "/03_train_valida_test_8/%s_x_y_train.csv"% name,index=False)
    x_y_valida_df.to_csv(base_dir + "/03_train_valida_test_8/%s_x_y_valida.csv"% name,index=False)
    x_y_test_df.to_csv(base_dir + "/03_train_valida_test_8/%s_x_y_test.csv"% name,index=False)

# train model
data_name = ["shanghai","shenzhen"]
for i in range(len(data_name)):
    name = data_name[i]
    x_y_train_df = pd.read_csv(base_dir + "/03_train_valida_test_8/%s_x_y_train.csv"% name)
    x_y_valida_df = pd.read_csv(base_dir + "/03_train_valida_test_8/%s_x_y_valida.csv"% name)

    x_y_train = x_y_train_df.values
    x_y_valida = x_y_valida_df.values

    len_t = x_y_train.shape[1]-8
    num_feats = 1

    input_layer = Input(batch_shape=(None,len_t,num_feats))
    model_list = [cbms.cnn_8,cbms.lstm_8,cbms.bilstm_8,cbms.cnn_lstm_8,cbms.cnn_bilstm_8,cbms.cbams_bilstm_8]
    output_layer_list = [model_list[i](input_layer) for i in range(len(model_list))]
    model_name = ["cnn_8","lstm_8","bilstm_8","cnn_lstm_8","cnn_bilstm_8","cbmams_bilstm_8"]

    for i in range(len(output_layer_list)):
        cbms.train_model(input_layer=input_layer,
                         output_layer=output_layer_list[i],
                         x_y_train=(x_y_train[:,:-8],x_y_train[:,-8:]),
                         x_y_valida=(x_y_valida[:,:-8],x_y_valida[:,-8:]),
                         loss_path=base_dir + "/04_loss_8/%s_%s_mse.png"% (name,model_name[i]),
                         model_path=base_dir + "/05_model_8/%s_%s.h5"% (name,model_name[i]))

# evaluate model
mcs = cbms.Mcstest()
mcs_results = []
data_name = ["shanghai","shenzhen"]
for i in range(len(data_name)):
    name = data_name[i]
    x_y_test_df = pd.read_csv(base_dir + "/03_train_valida_test_8/%s_x_y_test.csv"% name)
    x_y_test = x_y_test_df.values
    len_t = x_y_test.shape[1]-8
    num_feats = 1
    # load model
    model_name = ["cnn_8","lstm_8","bilstm_8","cnn_lstm_8","cnn_bilstm_8","cbmams_bilstm_8"]
    model_list = []
    pred_y_dict = {}
    metrics_results_list = []

    for j in range(len(model_name)):
        model = load_model(base_dir + "/05_model_8/%s_%s.h5"% (name,model_name[j]))
        # predict
        pred_y = model.predict(x_y_test[:,:-8].reshape((-1,len_t,num_feats)))
        pred_y_dict[model_name[j]] = pred_y
        # evaluate
        metrics_results,metrics_name = cbms.evaluate_model(x_y_test[:,-8:].flatten(),pred_y.flatten())
        metrics_results_list.append([model_name[j]]+metrics_results)
    #save metrics
    results_df = pd.DataFrame(metrics_results_list,columns=["model"]+metrics_name)
    results_df.to_csv(base_dir + "/06_results_8/%s_metrics.csv"% name,index=False)
    # compute mcs test
    error_list,error_name = mcs.error(real_y=x_y_test[:,-8:],pred_y=pred_y_dict)
    mcs_results.append([i.reshape((-1,i.shape[0],i.shape[1])) for i in error_list])

# compute mean
# transform dataframe to array
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
mcs_df = pd.DataFrame(mcs_dict,index=model_name)
mcs_df.to_csv(base_dir + "/06_results_8/mcs.csv",index=True)

