# A novel CBAMs-BiLSTM model for Chinese stock market forecasting

<p align="center">
  <img src="./paper/graphic abstract/graphic abstract.png" alt="graphic abstract">
</p>

## Abstract: 

The convolutional block attention module (CBAM) has demonstrated its superiority in various prediction problems, as it effectively enhances the prediction accuracy of deep learning models. However, there has been limited research testing the effectiveness of CBAM in predicting stock indexes. To fill this gap and improve the prediction accuracy of stock indexes, we propose a novel model called CBAMs-BiLSTM, which combines multiple CBAM modules with a bidirectional long short-term memory network (BiLSTM). In this study, we employ the standard metric evaluation method (SME) and the model confidence set test (MCS) to comprehensively evaluate the superiority and robustness of our model. We utilize two representative Chinese stock index data sets, namely, the SSE Composite Index and the SZSE Composite Index, as our experimental data. The numerical results demonstrate that CBAMs-BiLSTM outperforms BiLSTM alone, achieving average reductions of 13.06%, 13.39%, and 12.48% in MAE, RMSE, and MAPE, respectively. These findings confirm that CBAM can effectively enhance the prediction accuracy of BiLSTM. Furthermore, we compare our proposed model with other popular models and examine the impact of changing data sets, prediction methods, and the size of the training set. The results consistently demonstrate the superiority and robustness of our proposed model in terms of prediction accuracy and investment returns.

## Keywords: 
stock index; prediction; BiLSTM; CBAM; MCS; SME

## files introduction
there are three prediction methods in this paper. they are autoregressive one-step prediction, autoregressive multistep prediction and multivariate one-step prediction.

01_raw_data: raw stock index  
02_goal_data: data for autoregressive one-step prediction and autoregressive multistep prediction  
02_goal_data_m: data for multivariate one-step prediction  
03_train_valida_test: train dataset, validation dataset and test dataset for autoregressive one-step prediction  
03_train_valida_test_8: train dataset, validation dataset and test dataset for autoregressive multistep prediction (8 steps)  
03_train_valida_test_m: train dataset, validation dataset and test dataset for multivariate one-step prediction  
04_loss: loss curve for autoregressive one-step prediction  
04_loss_8: loss curve for autoregressive multistep prediction (8 steps)  
04_loss_m: loss curve for multivariate one-step prediction  
05_model: model (h5 file) for autoregressive one-step prediction  
05_model_8: model (h5 file) for autoregressive multistep prediction (8 steps)  
05_model_m: model (h5 file) for multivariate one-step prediction  
06_results: results (csv file) for autoregressive one-step prediction  
06_results_8: results (csv file) for autoregressive multistep prediction (8 steps)  
06_results_m: results (csv file) for multivariate one-step prediction  
07_paper: paper  
08_codes: codes for this paper  
main_libraries.txt: main libraries  

