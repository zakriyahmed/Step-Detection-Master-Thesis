import torch
import torch.nn as nn
from copy import deepcopy as dp

def get_prediction(model,data,ws,channel):
    labelpred = torch.zeros((data.shape[0],2))
    for i in range(0,int(labelpred.shape[0])-ws,ws):
        preds,prede = model(data[i:i+ws].T.float().reshape(1,channel,ws))
        preds,prede = nn.Sigmoid()(preds),nn.Sigmoid()(prede)

        preds[preds>=0.5]=1
        preds[preds<0.5]=0
        prede[prede>=0.5]=1
        prede[prede<0.5]=0

        labelpred[i:i+ws,0]= preds
        labelpred[i:i+ws,1]= prede
        
    return labelpred

def get_predictionLSTM(model,data,ws,channel):
    labelpred = torch.zeros((data.shape[0],2))
    for i in range(0,int(labelpred.shape[0])-ws,ws):
        pred = model(data[i:i+ws].float().reshape(1,ws,channel))
        preds,prede = nn.Sigmoid()(pred[0,0:ws]),nn.Sigmoid()(pred[0,ws:ws+ws])
        #print(pred.shape)
        preds[preds>=0.5]=1
        preds[preds<0.5]=0
        prede[prede>=0.5]=1
        prede[prede<0.5]=0

        labelpred[i:i+ws,0]= preds
        labelpred[i:i+ws,1]= prede
        
    return labelpred

def find_midpoints(relax_predictions):
    relaxmidpoints = torch.zeros_like(relax_predictions)
    mss=0
    mse=0
    mes=0
    mee=0
    for i in range(relaxmidpoints.shape[0]):
        if relax_predictions[i,0]==1 and relax_predictions[i+1,0]==1:
            if mss==0:
                mss=i

        if relax_predictions[i,0]==1 and relax_predictions[i+1,0]==0:
            mse=i

            if mse-mss>15:
                relaxmidpoints[int((mse+mss)/2),0]=1
            mss=0
            
    for i in range(relaxmidpoints.shape[0]):
        if relax_predictions[i,1]==1 and relax_predictions[i+1,1]==1:
            if mes==0:
                mes=i

        if relax_predictions[i,1]==1 and relax_predictions[i+1,1]==0:
            mee=i

            if mee-mes>15:
                relaxmidpoints[int((mee+mes)/2),1]=1
            mes=0

    
    return relaxmidpoints

def indexes(labels):
    return torch.where(labels[:,0]==1)[0],torch.where(labels[:,1]==1)[0]

def avg_distance(target,pred,threshold):
    mse =0
    count=0
    total = target.shape[0]
    histogram = torch.zeros(threshold)
    for i in range(pred.shape[0]):
        #print(i)
        for j in range(i,target.shape[0]):
            #print(j)
            if abs(pred[i]-target[j])<threshold:
                #print(relaxpred_index[i],target_index[j],relaxpred_index[i]-target_index[j])
                count+=1
                dis = abs(pred[i]-target[j])
                mse += dis
                histogram[dis.item()]+=1
            if abs(pred[i]-target[j])>32000:
                #print(j)
                #print(relaxpred_index[i],target_index[j],abs(relaxpred_index[i]-target_index[j]))
                break
        #print(i,j)

    return mse/count,count/total,count,histogram

def find_midpoints(relax_predictions,th):
    relaxmidpoints = torch.zeros_like(relax_predictions)
    mss=0
    mse=0
    mes=0
    mee=0
    for i in range(relaxmidpoints.shape[0]):
        if relax_predictions[i,0]==1 and relax_predictions[i+1,0]==1:
            if mss==0:
                mss=i

        if relax_predictions[i,0]==1 and relax_predictions[i+1,0]==0:
            mse=i

            if mse-mss>th:
                relaxmidpoints[int((mse+mss)/2),0]=1
            mss=0
            
    for i in range(relaxmidpoints.shape[0]):
        if relax_predictions[i,1]==1 and relax_predictions[i+1,1]==1:
            if mes==0:
                mes=i

        if relax_predictions[i,1]==1 and relax_predictions[i+1,1]==0:
            mee=i

            if mee-mes>th:
                relaxmidpoints[int((mee+mes)/2),1]=1
            mes=0

    
    return relaxmidpoints

def get_yolo_prediction(model,TestData,ch,ws,ss,g,x_size=39,w_size=53):
    predtestlabels = torch.zeros((TestData.shape[0],2))
    model= model.cpu().eval()
    with torch.no_grad():
        for i in range(0,TestData.shape[0]-ws,ss):
            pred = model(TestData[i:i+ws].T.reshape((1,ch,ws)).float())
            #pred = torch.round(pred)
            conf = torch.round(pred[:,:,2])
            confiscore = dp(conf)
            #print(i,pred[:,:,0]*39,pred[:,:,1]*53)
            #conf[conf>=0.5]=1
            x = pred[:,:,0]*x_size
            w = pred[:,:,1]*w_size
            #print(x.shape,w.shape,conf.shape)
            startp = torch.round(((x+g.T)-w)*conf)[0]
            #print(startp)
            endp = torch.round(((x+g.T)+w)*conf)[0]
            s = startp[startp.nonzero()][:,0].int()
            e = endp[endp.nonzero()][:,0].int()
            #print(i,s+i,e+i)
            #break
            predtestlabels[(s+i).tolist(),0]+=1
            predtestlabels[(e+i).tolist(),1]+=1
    return predtestlabels
      
            
def count_imbalance(label,device):
    one_S = torch.count_nonzero(label[:,0])
    one_E = torch.count_nonzero(label[:,1])

    ws1 = label[:,0].shape[0]/(2*one_S).to(torch.device(device))
    we1 = label[:,1].shape[0]/(2*one_E).to(torch.device(device))
    ws0 = label[:,0].shape[0]/(2*(label[:,0].shape[0]-one_S)).to(torch.device(device))
    we0 = label[:,1].shape[0]/(2*(label[:,0].shape[0]-one_E)).to(torch.device(device))
    
    return ws1,we1,ws0,we0