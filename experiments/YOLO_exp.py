import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split as tts
import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy as dp
import time 
import matplotlib.pyplot as plt
import dataloader as dt
import allmodel as am
from scipy import signal as S
from torcheval.metrics.functional import multiclass_f1_score as f1
import utils
import train
import datetime
import sys

log_file_name = "YOLO_log.txt"

# Redirect stdout to the log file
with open(log_file_name, "w") as log_file:
    sys.stdout = log_file

    print('Libraries loaded\n',flush=True)

    window_sizes = [200]  #1
    Gyro_settings = [True,False]  #2
    filter_settings = [True]#,False]  #1
    magnitude = [True,False]  #2
    Torelax = [False]  #1
    sensor_settings = [True,False] #2
    hand_test = [102091*2,102091*3]
    allsensors_test = [102091*0,102091*4]
    hand_train = [722582*2,722582*3]
    allsensors_train = [722582*0,722582*4]
    g=torch.tensor([0,40,80,120,160,0,40,80,120,160])

    out = {'Window_size':[],'Torelax':[],'Gyro':[],'Filter':[],'magnitude':[],'justHand':[],'F1s':[],'F1e':[],'MAE_S':[],'ratio_S':[],'hist_S':[],'count_S':[],'MAE_E':[],'ratio_E':[],'hist_E':[],'count_E':[],'total_steps':[]}
    count=0
    for ws in window_sizes:
        for justHand in sensor_settings:
            d = dt.Dataloader(ws,1)
            d.split(['Person (1)','Person (2)','Person (3)'])
            d.load()
            d.binary_labels()
            _,_,label,_,_,_,_,_,_,testlabel,_,_,_,_ = d.fetch(10,ws=ws,Gyro=False,ToFilter=False,magnitude=False)
            if justHand==True:                                
                train_here,train_there = hand_train[0],hand_train[1]
                test_here,test_there = hand_test[0],hand_test[1]
                yoloLabels = dt.yoloLabeler(label[train_here:train_there])
                yolotestlabels = dt.yoloLabeler(testlabel[test_here:test_there])
                T_s_index,T_e_index = utils.indexes(testlabel[hand_test[0]:hand_test[1]])
            elif justHand==False:
                train_here,train_there = allsensors_train[0],allsensors_train[1]
                test_here,test_there = allsensors_test[0],allsensors_test[1]
                yoloLabels = dt.yoloLabeler(label[train_here:train_there]) 
                yolotestlabels = dt.yoloLabeler(testlabel[test_here:test_there])
                T_s_index,T_e_index = utils.indexes(torch.tensor(yolotestLabels)[allsensors_test[0]:allsensors_test[1]])
        
            for gyro in Gyro_settings:
                for mag in magnitude:
                    for filtr in filter_settings:
                        for relax in Torelax:

                            Data,DataC,label,_,labelC,_,_,TestData,TestDataC,testlabel,testlabelC,_,_,_ = d.fetch(10,
                                                                                                                  ws=ws,Gyro=gyro,
                                                                                                                  ToFilter=filtr,
                                                                                                                  magnitude=mag)

                            print('Data loaded')

                            channel = DataC.shape[1]
                            batch_size=1024
                    # Create an instance of the LSTM network
                            model = am.yolo(ws,channel,(1,channel,ws)).cuda()

                            if relax==False:
                                dataset = torch.utils.data.TensorDataset(DataC[train_here:train_there].float().cuda(),
                                                                          torch.tensor(yoloLabels).float().cuda())
                                



                            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

                            model,_ =  train.trainYolo(model,dataloader,epochs=50,lr=0.001)

                            #testpredictionwoc = utils.get_yolo_prediction(model.eval().cpu(),TestData[test_here:test_there],ws,channel)

                            
                            if relax==False:
                                testpredictionwoc = utils.get_yolo_prediction(model,TestData[test_here:test_there],
                                                                           channel,ws,ws,g,x_size=39,w_size=53)
                                f1s = f1(testlabel[test_here:test_there,0].cpu(),testpredictionwoc[:,0],num_classes=2)
                                f1e = f1(testlabel[test_here:test_there,1].cpu(),testpredictionwoc[:,1],num_classes=2)
                                P_s_index,P_e_index = utils.indexes(testpredictionwoc)
                                





                            maeS,percentageS,countS,histS = utils.avg_distance(T_s_index,P_s_index,10)
                            maeE,percentageE,countE,histE = utils.avg_distance(T_e_index,P_e_index,10)

                            out['Window_size'].append(ws)
                            out['Torelax'].append(relax)
                            out['Gyro'].append(gyro)
                            out['Filter'].append(filtr)
                            out['justHand'].append(justHand)
                            out['F1s'].append(f1s.item())
                            out['F1e'].append(f1e.item())
                            out['MAE_S'].append(maeS.item())
                            out['ratio_S'].append(percentageS)
                            out['hist_S'].append(histS.tolist())
                            out['count_S'].append(countS)
                            out['MAE_E'].append(maeE.item())
                            out['ratio_E'].append(percentageE)
                            out['hist_E'].append(histE.tolist())
                            out['count_E'].append(countE)
                            out['total_steps'].append(T_s_index.shape[0])
                            out['magnitude'].append(mag)

                            now = datetime.datetime.now()
                            for_now = now.strftime("%d.%m.%Y %H:%M:%S")
                            #with open("CNN_log.txt", "w") as text_file:
                            print('\n',count,'\t',for_now,'\n',flush=True)
                            print(out,flush=True)
                            print('\n\n',flush=True)
                            count+=1

                            del dataset,dataloader,model,DataC,TestDataC
        
result = pd.DataFrame(out)
result.to_csv('YOLO.csv')
        
        
