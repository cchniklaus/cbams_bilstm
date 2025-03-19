# coding=utf-8
# author: Chenhao Cui
# date:20250220
# email: 3313398924@qq.com
# description: modules of paper "A novel CBAMs-BiLSTM model for Chinese stock market forecasting"
# version: tensorflow.keras without gpu

# library
import random as rn
from typing import List, Union, Tuple, Dict
import pandas as pd
import numpy as np
from scipy.stats import jarque_bera
from statsmodels.stats.diagnostic import acorr_ljungbox
import tensorflow as tf
from keras.models import Model
from keras.layers import *
from keras.optimizers import Adam
from keras import backend
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
import matplotlib as mpl
import matplotlib.pyplot as plt
from arch.bootstrap import MCS
mpl.use('TkAgg')
warnings.filterwarnings("ignore")

# modules

def repetition(x=123, y=12345, z=1234):
    """
    
    """
    np.random.seed(x)
    rn.seed(y)
    session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    tf.random.set_seed(z)
    sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
    tf.compat.v1.keras.backend.set_session(sess)
    return None

def compute_statistics(data_df: pd.DataFrame,statistics_list: List=["count","mean","min","max","std","jb_pvalue","lb_pvalue"]) -> pd.DataFrame:
    """
    according to data_df and statistics_list, compute statistics.
    param data_df: a pd.DataFrame (df) by which statistics will be computed and a column of this df represents a sample.
    param statistics_list: a list that contains goal statistics name.
    return: a pd.DataFrame that contains the statistics value corresponding to statistics_list.
    """
    default_statistics_list = ["count","mean","min","max","std","jb_pvalue","lb_pvalue"]
    results_dict = {}
    if set(statistics_list).issubset(default_statistics_list):
        n,m = data_df.shape
        name = list(data_df.columns)
        count_array = np.array([n]*m).reshape((-1,1))
        results_dict["count"] = count_array
        mean_array = data_df.mean(axis=0).values.reshape((-1,1))
        results_dict["mean"] = mean_array
        min_array = data_df.min(axis=0).values.reshape((-1,1))
        results_dict["min"] = min_array
        max_array = data_df.max(axis=0).values.reshape((-1,1))
        results_dict["max"] = max_array
        std_array = data_df.std(axis=0).values.reshape((-1,1))
        results_dict["std"] = std_array
        jb_pvalue_array = np.array([jarque_bera(data_df.values[:,i]).pvalue for i in range(m)]).reshape((-1,1))
        results_dict["jb_pvalue"] = jb_pvalue_array
        lb_pvalue_array = np.array([acorr_ljungbox(data_df.values[:,i],lags=20)["lb_pvalue"].iloc[-1] for i in range(m)]).reshape((-1,1))
        results_dict["lb_pvalue"] = lb_pvalue_array
        
        statistics = [results_dict[x] for x in statistics_list]
        statistics_array = np.concatenate([np.array(name).reshape((-1,1))] + statistics,axis=1)
        cols = ["file_name"] + statistics_list
        statistics_df = pd.DataFrame(statistics_array,columns=cols)
        return statistics_df
    else:
        print("default statistics are", default_statistics_list)
        raise ValueError("please check it carefully.")

def transform_data(data_list: List, dim_x: int, dim_y: int=1) -> np.ndarray:
    """
    transform data with shape (n,) to data with shape (dim_x,dim_y) for training model.
    param data_list: a list that contains data to be transformed.
    param dim_x: int, the dimension of features.
    param dim_y: int, the dimension of target, the default value is 1.
    return: np.ndarray.
    """
    x, y = [], []
    n = len(data_list)
    step = 1
    m = int(np.ceil((n-(dim_x + dim_y))/step)) + 1
    for i in range(m):
        windows = data_list[i:i+dim_x+dim_y]
        x.append(windows[:dim_x])
        y.append(windows[dim_x:])
    x_y = np.concatenate([np.array(x),np.array(y)],axis=1)
    return x_y

##################### CBAM1D
channel_axis = 1 if backend.image_data_format() == "channels_first" else 2

# CAM
def channel_attention(input_layer: tf.Tensor, reduction_ratio: float=0.125) -> tf.Tensor:
    """
    the structure of CAM.
    param input_layer: input layer.
    param reduction_ratio: reduction ratio.
    return: tf.Tensor
    """
    # get channel
    channel = int(input_layer.shape[channel_axis])
    maxpool_channel = GlobalMaxPooling1D()(input_layer)
    maxpool_channel = Reshape((1, channel))(maxpool_channel)
    avgpool_channel = GlobalAvgPool1D()(input_layer)
    avgpool_channel = Reshape((1, channel))(avgpool_channel)
    Dense_One = Dense(units=int(channel * reduction_ratio), activation='relu', kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')
    Dense_Two = Dense(units=int(channel), activation='relu', kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')
    # max path
    mlp_1_max = Dense_One(maxpool_channel)
    mlp_2_max = Dense_Two(mlp_1_max)
    mlp_2_max = Reshape(target_shape=(1, int(channel)))(mlp_2_max)
    # avg path
    mlp_1_avg = Dense_One(avgpool_channel)
    mlp_2_avg = Dense_Two(mlp_1_avg)
    mlp_2_avg = Reshape(target_shape=(1, int(channel)))(mlp_2_avg)
    channel_attention_feature = Add()([mlp_2_max, mlp_2_avg])
    channel_attention_feature = Activation('sigmoid')(channel_attention_feature)
    return Multiply()([channel_attention_feature, input_layer])

# SAM
def spatial_attention(channel_refined_feature: tf.Tensor) -> tf.Tensor:
    """
    the structure of SAM.
    param channel_refined_feature: tf.Tensor
    return: tf.Tensor.
    """
    maxpool_spatial = Lambda(lambda x: backend.max(x, axis=2, keepdims=True))(channel_refined_feature)
    avgpool_spatial = Lambda(lambda x: backend.mean(x, axis=2, keepdims=True))(channel_refined_feature)
    max_avg_pool_spatial = Concatenate(axis=2)([maxpool_spatial, avgpool_spatial])
    return Conv1D(filters=1, kernel_size=3, padding="same", activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(max_avg_pool_spatial)

def cbam_module(input_layer: tf.Tensor, reduction_ratio: float=0.5) -> tf.Tensor:
    """
    the structure of CBAM.
    param input_layer: input layer.
    param reduction_ratio: reduction ratio.
    return: tf.Tensor
    """
    channel_refined_feature = channel_attention(input_layer, reduction_ratio=reduction_ratio)
    spatial_attention_feature = spatial_attention(channel_refined_feature)
    refined_feature = Multiply()([channel_refined_feature, spatial_attention_feature])
    return Add()([refined_feature, input_layer])

# bilstm
def bilstm(input_layer: tf.Tensor) -> tf.Tensor:
    """
    the structure of bilstm.
    param input_layer: input layer.
    return: tf.Tensor
    """
    layer1 = Bidirectional(LSTM(units=64,use_bias=True,return_sequences=False))(input_layer)
    layer2 = Dense(1,use_bias=True,activation="sigmoid")(layer1)
    return layer2

# bilstm_cbam
def bilstm_cbam(input_layer: tf.Tensor) -> tf.Tensor:
    """
    the structure of bilstm_cbam.
    param input_layer: input layer.
    return: tf.Tensor
    """
    layer1 = Bidirectional(LSTM(units=64,use_bias=True,return_sequences=True))(input_layer)
    layer2 = cbam_module(layer1)
    layer3 = Flatten()(layer2)
    layer4 = Dense(1,use_bias=True,activation="sigmoid")(layer3)
    return layer4

# cbams_bilstm
def cbams_bilstm(input_layer: tf.Tensor,cbam_n: int) -> tf.Tensor:
    """
    the structure of cbams_bilstm.
    param input_layer: input layer.
    param cbam_n: number of cbam.
    return: tf.Tensor.
    """
    cbam = cbam_module(input_layer)
    if cbam_n > 1:
        for _ in range(cbam_n-1):
            cbam = cbam_module(cbam)
    layer1 = Bidirectional(LSTM(64,use_bias=True,return_sequences=False))(cbam)
    layer2 = Dense(1,use_bias=True,activation="sigmoid")(layer1)
    return layer2

# plot loss
def plot_loss(train_loss: np.ndarray,valida_loss: np.ndarray, fig_path: str) -> None:
    """
    plot loss.
    param train_loss: np.ndarray, train loss.
    param valida_loss: np.ndarray, validation loss.
    param fig_path: str, the path of figure
    return: None.
    """
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.figure(figsize=(14,6))
    plt.plot(train_loss,label="train loss",color="blue")
    plt.plot(valida_loss,label="validation loss",color="red")
    plt.xlabel("Epochs",fontsize=12)
    plt.ylabel("Loss",fontsize=12)
    plt.title("Loss Curve",fontsize=12)
    plt.legend()
    plt.savefig(fig_path)
    plt.close()
    return None

# evaluate model
def evaluate_model(real_y: np.ndarray,pred_y:np.ndarray) -> Tuple:
    """
    evaluate model.
    param real_y: np.ndarray, real y.
    param pred_y: np.ndarray, predicted y.
    return: Tuple,(results_list,name_list).
    """
    mae = mean_absolute_error(real_y,pred_y)
    rmse = mean_squared_error(real_y,pred_y)**0.5
    r2 = r2_score(real_y,pred_y)
    mape=np.mean(np.abs(real_y-pred_y)/(np.abs(real_y)+10**(-6)))
    return [mae,rmse,r2,mape],["mae","rmse","r2","mape"]

# mcs_test
class Mcstest:
    """
    mcs test.

    Methods:
    error: compute error.
    mcs_test: mcs test.
    """
    def __init__(self):
        pass

    def error(self,real_y: np.ndarray, pred_y: Dict) -> Tuple:
        """
        compute error.
        param real_y: np.ndarray, real y.
        param pred_y: Dict, predicted y. the keys of the dict is model name and the values of the dict is predicted y.
        return: Tuple.
        """
        model_names = list(pred_y.keys())
        r = real_y.shape
        p = pred_y[list(pred_y.keys())[0]].shape
        if len(model_names) < 2 :
            raise ValueError("Amount for models has to greater than 1.")
        if r != p:
            raise ValueError("The shape of real_y and element of pred_y have to be the same.")
        mse_dict = {}
        mae_dict = {}
        hmse_dict = {}
        hmae_dict = {}
        for i in range(len(model_names)):
            name = model_names[i]
            y_ = pred_y[name]
            # mse error
            mse_error = (real_y-y_)**2 if len(r) == 1 else np.sum((real_y-y_)**2,axis=1)
            mse_dict[name] = mse_error
            # mae error
            mae_error = np.abs(real_y-y_) if len(r) == 1 else np.sum(np.abs(real_y-y_),axis=1)
            mae_dict[name] = mae_error
            # hmse error
            hmse_error = (np.abs(real_y-y_)/(np.abs(real_y)+10**(-6)))**2 if len(r) == 1 else np.sum((np.abs(real_y-y_)/(np.abs(real_y)+10**(-6)))**2,axis=1)
            hmse_dict[name] = hmse_error
            # hmae error
            hmae_error = np.abs(real_y-y_)/(np.abs(real_y)+10**(-6)) if len(r) == 1 else np.sum(np.abs(real_y-y_)/(np.abs(real_y)+10**(-6)),axis=1)
            hmae_dict[name] = hmae_error
        mse_ = pd.DataFrame(mse_dict).values
        mae_ = pd.DataFrame(mae_dict).values
        hmse_ = pd.DataFrame(hmse_dict).values
        hmae_ = pd.DataFrame(hmae_dict).values
        return [mse_,mae_,hmse_,hmae_],["mse","mae","hmse","hmae"]

    def mcs_test(self,error,statistic:str) -> pd.Series:
        """
        mcs test.
        param error: dataframe.
        param statistic: str, the name of statistic, R or MAX.
        return: pd.Series.
        """
        if statistic not in ["R","MAX"]:
            raise ValueError("please check the statistic, and alternatives are R and MAX")
        else:
            mcs=MCS(error,size=0.05,method=statistic,seed=12345)
            mcs.compute()
            return mcs.pvalues.values

# train_model
def train_model(input_layer: tf.Tensor, 
                output_layer: tf.Tensor, 
                x_y_train: Tuple, 
                x_y_valida: Tuple, 
                loss_path: str, 
                model_path: str) -> None:
    """
    train model.
    param input_layer: tf.Tensor, input layer.
    param output_layer: tf.Tensor, output layer.
    param x_y_train: tuple, training data.
    param x_y_valida: tuple, validation data.
    param loss_path: str, the path of loss figure.
    param model_path: str, the path of model.
    return: None.
    """
    repetition(12,1234,2345)
    m = Model(inputs=input_layer,outputs=output_layer)
    m.compile(loss='mse',optimizer='adam',metrics=['mse'])
    m.fit(x_y_train[0],
          x_y_train[1],
          shuffle=True,
          verbose=1,
          validation_data=(x_y_valida[0],x_y_valida[1]),
          batch_size=128,epochs=200)
    
    train_mse = m.history.history['mse']
    valida_mse = m.history.history['val_mse']
    plot_loss(train_mse,valida_mse,loss_path)
    m.save(model_path)
    return None

def cnn(layer: tf.Tensor) -> tf.Tensor:
    """
    structure of cnn.
    param layer: tf.Tensor.
    return: tf.Tensor.
    """
    c = Conv1D(filters=64,kernel_size=3)(layer)
    m = MaxPool1D(pool_size=2)(c)
    f = Flatten()(m)
    d = Dense(units=1,use_bias=True,activation="sigmoid")(f)
    return d

def lstm(layer: tf.Tensor) -> tf.Tensor:
    """
    structure of lstm.
    param layer: tf.Tensor.
    return: tf.Tensor.
    """
    l = LSTM(units=64,use_bias=True,return_sequences=False)(layer)
    d = Dense(units=1,use_bias=True,activation="sigmoid")(l)
    return d

def cnn_lstm(layer: tf.Tensor) -> tf.Tensor:
    """
    structure of cnn_lstm.
    param layer: tf.Tensor.
    return: tf.Tensor.
    """
    c = Conv1D(filters=64,kernel_size=3)(layer)
    m = MaxPool1D(pool_size=2)(c)
    l = LSTM(64, use_bias=True, return_sequences=False)(m)
    d = Dense(units=1,use_bias=True,activation="sigmoid")(l)
    return d

def cnn_bilstm(layer: tf.Tensor) -> tf.Tensor:
    """
    structure of cnn_bilstm.
    param layer: tf.Tensor.
    return: tf.Tensor.
    """
    c = Conv1D(filters=64,kernel_size=3)(layer)
    m = MaxPool1D(pool_size=2)(c)
    l = Bidirectional(LSTM(64, use_bias=True, return_sequences=False))(m)
    d = Dense(units=1,use_bias=True,activation="sigmoid")(l)
    return d

def cnn_8(layer: tf.Tensor) -> tf.Tensor:
    """
    structure of cnn.
    param layer: tf.Tensor.
    return: tf.Tensor.
    """
    c = Conv1D(filters=64,kernel_size=3)(layer)
    m = MaxPool1D(pool_size=2)(c)
    f = Flatten()(m)
    d = Dense(units=8,use_bias=True,activation="sigmoid")(f)
    return d

def lstm_8(layer: tf.Tensor) -> tf.Tensor:
    """
    structure of lstm.
    param layer: tf.Tensor.
    return: tf.Tensor.
    """
    l = LSTM(units=64,use_bias=True,return_sequences=False)(layer)
    d = Dense(units=8,use_bias=True,activation="sigmoid")(l)
    return d

def bilstm_8(input_layer: tf.Tensor) -> tf.Tensor:
    """
    the structure of bilstm.
    param input_layer: input layer.
    return: tf.Tensor
    """
    layer1 = Bidirectional(LSTM(units=64,use_bias=True,return_sequences=False))(input_layer)
    layer2 = Dense(8,use_bias=True,activation="sigmoid")(layer1)
    return layer2

def cnn_lstm_8(layer: tf.Tensor) -> tf.Tensor:
    """
    structure of cnn_lstm.
    param layer: tf.Tensor.
    return: tf.Tensor.
    """
    c = Conv1D(filters=64,kernel_size=3)(layer)
    m = MaxPool1D(pool_size=2)(c)
    l = LSTM(64, use_bias=True, return_sequences=False)(m)
    d = Dense(units=8,use_bias=True,activation="sigmoid")(l)
    return d

def cnn_bilstm_8(layer: tf.Tensor) -> tf.Tensor:
    """
    structure of cnn_bilstm.
    param layer: tf.Tensor.
    return: tf.Tensor.
    """
    c = Conv1D(filters=64,kernel_size=3)(layer)
    m = MaxPool1D(pool_size=2)(c)
    l = Bidirectional(LSTM(64, use_bias=True, return_sequences=False))(m)
    d = Dense(units=8,use_bias=True,activation="sigmoid")(l)
    return d

def cbams_bilstm_8(input_layer: tf.Tensor) -> tf.Tensor:
    """
    the structure of cbams_bilstm.
    param input_layer: input layer.
    return: tf.Tensor.
    """
    cbam_n = 15
    cbam = cbam_module(input_layer)
    if cbam_n > 1:
        for _ in range(cbam_n-1):
            cbam = cbam_module(cbam)
    layer1 = Bidirectional(LSTM(64,use_bias=True,return_sequences=False))(cbam)
    layer2 = Dense(8,use_bias=True,activation="sigmoid")(layer1)
    return layer2

def transform_data_m(data: np.ndarray, dim_x: int, dim_y: int=1) -> Tuple:
    """
    transform data with shape (n,s) to feats with dim_x and labels with dim_y for training model.
    param data: np.ndarray, data.
    param dim_x: int, the dimension of features.
    param dim_y: int, the dimension of target, the default value is 1.
    return: Tuple.
    """
    x, y = [], []
    index_n_samples = len(data) - 1 - (dim_x + dim_y - 1) 
    for i in range(index_n_samples + 1):
        x.append(data[0 + i:dim_x + i])
        y.append(data[dim_x + i:dim_x + dim_y + i,3])
    return x, y

