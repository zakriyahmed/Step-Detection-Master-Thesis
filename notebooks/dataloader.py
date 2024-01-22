import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split as tts
import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy as dp
import time 
from scipy import signal as S

class Dataloader():
    def __init__(self,window_size,step_size,split_ratio=0.3,shuffle=False,activity=False):
        self.window_size = window_size
        self.step_size = step_size
        self.split_ratio = split_ratio
        self.shuffle = shuffle
        self.record = {'Test':[],'window_size':[],'Accuracy_S':[],'TP_S':[],'TN_S':[],
                       'FP_S':[],'FN_S':[],'Accuracy_E':[],'TP_E':[],
                       'TN_E':[],'FP_E':[],'FN_E':[],'Minimums':[],'Maximums':[]}
                
        self.datafiles = []
        self.labelfiles = []
        self.name = []
        self.IsActivity = activity
        self.markers = {}
        good = np.sort(os.listdir('good'))
        

        #create a list of columns names to be used for loading each value
        #self.col_names = [' AccelX_5', ' AccelY_5', ' AccelZ_5', ' GyroX_5', ' GyroY_5', ' GyroZ_5',]
        # use all sensors except feet
        self.col_names = [[f' AccelX_{i}', f' AccelY_{i}', f' AccelZ_{i}', f' GyroX_{i}', f' GyroY_{i}', f' GyroZ_{i}'] for i in range(3,7)]
        self.col_names = [item for sublist in self.col_names for item in sublist]
        if activity:
            self.col_names.append(' Activity')
        # create an empty dataframe to load all files into it, i.e. concat each file at the end of previous
        self.traindata = pd.DataFrame(columns =self.col_names)
        self.testdata = pd.DataFrame(columns =self.col_names)
        #colums names for label files
        self.col_names_label = ['start','end']
        #empty dataframe for labels
        self.trainindexlabels = pd.DataFrame(columns = self.col_names_label)
        self.testindexlabels = pd.DataFrame(columns = self.col_names_label)
        
        for folder in good:
            subfolder = np.sort(os.listdir(f'good/{folder}'))
            #check each file in the folder
            
            for files in sorted(subfolder):
                #if its a csv file, i.e. data file
                if files.endswith('.csv'):
                    self.datafiles.append(f'good/{folder}/{files}')
                    self.name.append(f'{folder}/{files}')
                    self.markers[f'good/{folder}/{files}'] = [folder]
                    
                if files.endswith('.csv.stepMixed'):
                    self.labelfiles.append(f'good/{folder}/{files}')   
                    
                
    def split(self,names):

        
        self.trainfiles = []
        self.trainlabels = []
        self.testfiles = []
        self.testlabels = []
        
        for file in self.datafiles:    
            for name in names:
            
                if name in file:
                    self.testfiles.append(file)
                    self.datafiles.remove(file)
                    del self.markers[file]
        for file in self.labelfiles:
            for name in names:
            
                if name in file:
                    self.testlabels.append(file)
                    self.labelfiles.remove(file)

        
        self.trainfiles = dp(self.datafiles)
        self.trainlabels = dp(self.labelfiles)
        

    def load(self,concat=True):
        # read files from memory and concat them.
        # concat = True to take the columns from different sensors and merge them all in acc_xyz and gyro_xyz total 6 colums,
        # else leave 24 columns as it.
        offset = 0
        for i in range(len(self.trainfiles)):
            d = pd.read_csv(self.trainfiles[i])
            l = pd.read_csv(self.trainlabels[i],names = self.col_names_label).sort_values('start').reset_index(drop=True)
            
            l = dp(l+offset)
            
            d = d[self.col_names]
            self.traindata = pd.concat((self.traindata,d),ignore_index=True)
            self.trainindexlabels = pd.concat((self.trainindexlabels,l),ignore_index=True)
            
            offset += len(d)
            self.markers[self.trainfiles[i]].append(offset)
            
        offset = 0
        for i in range(len(self.testfiles)):
            dt = pd.read_csv(self.testfiles[i])
            lt = pd.read_csv(self.testlabels[i],names = self.col_names_label).sort_values('start').reset_index(drop=True)           
            lt = dp(lt+offset)
            dt = dt[self.col_names]
            self.testdata = pd.concat((self.testdata,dt),ignore_index=True)
            
            self.testindexlabels = pd.concat((self.testindexlabels,lt),ignore_index=True)
            
            offset += len(dt)
        if self.IsActivity:    
            self.train_activity = self.traindata[' Activity']
            self.test_activity = self.testdata[' Activity']
        if concat:
            single_sensor = self.traindata.shape[0]
            tsingle_sensor = self.testdata.shape[0]
            sensor_labels = np.zeros((single_sensor*4))
            tsensor_labels = np.zeros((tsingle_sensor*4))
            for i in range(1,4):
                sensor_labels[single_sensor*i:single_sensor*(i+1)] = i
                tsensor_labels[tsingle_sensor*i:tsingle_sensor*(i+1)] = i
            newnames=[' AccelX_', ' AccelY_',' AccelZ_', ' GyroX_', ' GyroY_', ' GyroZ_']
            new = pd.DataFrame()
            for col in newnames:
                col_to_merge = [col+'3', col+'4', col+'5', col+'6']
                new[col] = pd.concat([self.traindata[col_to_merge[0]],
                                      self.traindata[col_to_merge[1]],
                                      self.traindata[col_to_merge[2]],
                                      self.traindata[col_to_merge[3]]])
            
            self.traindata = dp(new)
            new = pd.DataFrame()
            for col in newnames:
                col_to_merge = [col+'3', col+'4', col+'5', col+'6']
                new[col] = pd.concat([self.testdata[col_to_merge[0]],
                                      self.testdata[col_to_merge[1]],
                                      self.testdata[col_to_merge[2]],
                                      self.testdata[col_to_merge[3]]])
            
            self.testdata = dp(new)
            self.sensor_labels = sensor_labels
            self.tsensor_labels = tsensor_labels
        #return self.traindata,self.testdata,self.trainindexlabels,self.testindexlabels
            
    def binary_labels(self,concat=True):
        #convert index labels to binary labels
        self.trainlabelB = np.zeros((int(len(self.traindata)/4),2),dtype=np.int8)
        self.testlabelB = np.zeros((int(len(self.testdata)/4),2),dtype=np.int8)
        
        for i in range(len(self.trainindexlabels)):
            self.trainlabelB[self.trainindexlabels['start'][i],0] = 1
            self.trainlabelB[self.trainindexlabels['end'][i],1] = 1
        #print(len(self.testdata),self.testlabel.shape,len(self.testindexlabels))    
        for i in range(len(self.testindexlabels)):
            #print(i,self.testindexlabels['start'][i])
            self.testlabelB[self.testindexlabels['start'][i],0] = 1
            self.testlabelB[self.testindexlabels['end'][i],1] = 1
        if concat:

            self.trainlabel = dp(np.concatenate((self.trainlabelB,self.trainlabelB,self.trainlabelB,self.trainlabelB)))
            self.testlabel = dp(np.concatenate((self.testlabelB,self.testlabelB,self.testlabelB,self.testlabelB)))

        #return self.trainlabel,self.testlabel
    
    def data(self,binary=True):
         #convert pd to numpy arrays and convert into windows
        self.load()
        D_tr = self.traindata.to_numpy().T.reshape((1,6,len(self.traindata)))
        D_te = self.testdata.to_numpy().T.reshape((1,6,len(self.testdata)))
        
        D_tr_s = np.zeros((int(D_tr.shape[2]/self.step_size),6,self.window_size))
        D_te_s = np.zeros((int(D_te.shape[2]/self.step_size),6,self.window_size))
        
        if binary:
            self.binary_labels()
            L_tr_s = np.zeros((int(self.trainlabel.shape[0]/self.step_size),2,self.window_size))
            L_te_s = np.zeros((int(self.testlabel.shape[0]/self.step_size),2,self.window_size))
            
            for i in range(int(D_tr_s.shape[0]/self.step_size)-self.window_size):
                D_tr_s[i] = D_tr[0,:,i*self.step_size:(i*self.step_size)+self.window_size]
                L_tr_s[i,0] = self.trainlabel[i*self.step_size:(i*self.step_size)+self.window_size,0]
                L_tr_s[i,1] = self.trainlabel[i*self.step_size:(i*self.step_size)+self.window_size,1]
                
            for i in range(int(D_te_s.shape[0]/self.step_size)-self.window_size):
                D_te_s[i] = D_te[0,:,i*self.step_size:(i*self.step_size)+self.window_size]
                L_te_s[i,0] = self.testlabel[i*self.step_size:(i*self.step_size)+self.window_size,0]
                L_te_s[i,1] = self.testlabel[i*self.step_size:(i*self.step_size)+self.window_size,1]
                
        return D_tr_s,D_te_s,L_tr_s,L_te_s
    
    def fetch(self,relax,ws,ToFilter=True,Gyro=False,magnitude=True,scale=True,Sclass=False,relaxlabels=False,chunked=True):
        # -------------------inputs--------------------------------------
        # relax ----- how much on left and right are the label relaxed
        # ws ------ window size, when dividing the signal into chunks
        #-------------------outputs--------------------------------------
        # 1. data ---- training signal without divided into windows
        # 2. traindata ---- training signal divided into windows of size ws
        # 3. label ----- original labels
        # 4. relaxlabel ---- labels relaxed by relax
        # 5. trainlabel---- original labels in 3 divided into windows
        # 6. trainlabelrelax ---- relaxlabel divided into windows
        # 7. testdata ----- test signal without divided into windows
        # 8. testdatac ---- test data divided into windows.
        # 9. testlabel ---- original labels for test set
        # 10. testlabelc ----- original labels for test divided into windows
        # 11. testrelaxlabel ----- original labels relaxed by relax
        # 12. testrelaxlabelc ----- 10 divided into windows
        
        if Gyro:
            no_columns=6
        else:
            no_columns=3
        
        if magnitude:
            if Gyro:
                no_columns+=2
            else:
                no_columns+=1
                
        
        #da,ta,trl,tel = self.data()
        data = self.traindata.to_numpy()
        testdata = self.testdata.to_numpy()
        #print(data.shape)
        if ToFilter:
            print('Filtering Signal')
            sos = S.butter(3, 3, 'lp', fs=80, output='sos')
            filteredax = S.sosfilt(sos, data[:,0])
            filtereday = S.sosfilt(sos, data[:,1])
            filteredaz = S.sosfilt(sos, data[:,2])
            tfilteredax = S.sosfilt(sos, testdata[:,0])
            tfiltereday = S.sosfilt(sos, testdata[:,1])
            tfilteredaz = S.sosfilt(sos, testdata[:,2])
            if magnitude:
                print('Adding magnitude as feature')
                filteredam = np.sqrt( filteredax**2 + filtereday**2 + filteredaz**2)
                tfilteredam = np.sqrt( tfilteredax**2 + tfiltereday**2 + tfilteredaz**2)
            if Gyro==True:
                print('Adding filtered Gyro sensor data')
                filteredgx = S.sosfilt(sos, data[:,3])
                filteredgy = S.sosfilt(sos, data[:,4])
                filteredgz = S.sosfilt(sos, data[:,5])
                tfilteredgx = S.sosfilt(sos, testdata[:,3])
                tfilteredgy = S.sosfilt(sos, testdata[:,4])
                tfilteredgz = S.sosfilt(sos, testdata[:,5])
                if magnitude:                    
                    filteredgm = np.sqrt( filteredgx**2 + filteredgy**2 + filteredgz**2)
                    tfilteredgm = np.sqrt( tfilteredgx**2 + tfilteredgy**2 + tfilteredgz**2)
            
            if magnitude and Gyro:
                filtered = np.concatenate(([[filteredax,filtereday,filteredaz,
                                             filteredgx,filteredgy,filteredgz,
                                             filteredam,filteredgm]])).T 
                tfiltered = np.concatenate(([[tfilteredax,tfiltereday,tfilteredaz,
                                              tfilteredgx,tfilteredgy,tfilteredgz,
                                              tfilteredam,tfilteredgm]])).T
            if magnitude and not Gyro:
                filtered = np.concatenate(([[filteredax,filtereday,filteredaz,filteredam]])).T
                tfiltered = np.concatenate(([[tfilteredax,tfiltereday,tfilteredaz,tfilteredam]])).T
            if not magnitude and Gyro:
                filtered = np.concatenate(([[filteredax,filtereday,filteredaz,filteredgx,filteredgy,filteredgz]])).T
                tfiltered = np.concatenate(([[tfilteredax,tfiltereday,tfilteredaz,tfilteredgx,tfilteredgy,tfilteredgz]])).T
            if not magnitude and not Gyro:
                filtered = np.concatenate(([[filteredax,filtereday,filteredaz]])).T
                tfiltered = np.concatenate(([[tfilteredax,tfiltereday,tfilteredaz]])).T
            
            data = torch.tensor(filtered).reshape((data.shape[0],no_columns))
            testdata = torch.tensor(tfiltered).reshape((tfiltered.shape[0],no_columns))

        if not ToFilter:
            print('Not Filtering Signal')
            if magnitude:
                print('Adding magnitude as feature')
                am = np.sqrt(data[:,0]**2 + data[:,1]**2 +data[:,2]**2)
                tam = np.sqrt(testdata[:,0]**2 + testdata[:,1]**2 + testdata[:,2]**2)
                if Gyro:
                    gm = np.sqrt(data[:,3]**2 + data[:,4]**2 +data[:,5]**2)
                    tgm = np.sqrt(testdata[:,3]**2 + testdata[:,4]**2 + testdata[:,5]**2)
            if Gyro and magnitude:
                print('Gyro and Magnitude')
                
                data = np.concatenate(([[data[:,0],data[:,1],data[:,2],data[:,3],data[:,4],data[:,5],am,gm]])).T
                testdata = np.concatenate(([[testdata[:,0],testdata[:,1],testdata[:,2],
                                             testdata[:,3],testdata[:,4],testdata[:,5],
                                             tam,tgm]])).T
            if not Gyro and magnitude:
                data = np.concatenate(([[data[:,0],data[:,1],data[:,2],am]])).T
                testdata = np.concatenate(([[testdata[:,0],testdata[:,1],testdata[:,2],tam]])).T
            if Gyro and not magnitude:
                data = np.concatenate(([[data[:,0],data[:,1],data[:,2],data[:,3],data[:,4],data[:,5]]])).T
                testdata = np.concatenate(([[testdata[:,0],testdata[:,1],testdata[:,2],testdata[:,3],testdata[:,4],testdata[:,5]]])).T
            if not Gyro and not magnitude:
                data = np.concatenate(([[data[:,0],data[:,1],data[:,2]]])).T
                testdata = np.concatenate(([[testdata[:,0],testdata[:,1],testdata[:,2]]])).T
            
            data = torch.tensor(data)
            testdata = torch.tensor(testdata)
        #print(f'Data shape : {data.shape}')
        #print(f'Test Data shape :{testdata.shape}')
            
        if scale:
            minimums = data.min(axis=0)
            maximums = data.max(axis=0)
            data=(data-minimums[0])/(maximums[0]-minimums[0])
            testdata = (testdata-minimums[0])/(maximums[0]-minimums[0])
        label = torch.tensor(self.trainlabel).reshape((self.trainlabel.shape[0],2))
        sensorClasses = torch.tensor(self.sensor_labels)
        
        relaxlabel = torch.clone(label)
        for i in range(relaxlabel.shape[0]):
            if label[i,0]==1:
                relaxlabel[i-relax:i+relax,0]=1
            if label[i,1]==1:
                relaxlabel[i-relax:i+relax,1]=1

        traindata = torch.zeros((int(data.shape[0]/self.step_size),no_columns,self.window_size))
        trainlabel = torch.zeros((int(label.shape[0]/self.step_size),2,self.window_size))
        trainlabelrelax = torch.zeros((int(relaxlabel.shape[0]/self.step_size),2,self.window_size))
        #sensorClasses = torch.zeros((int(Slabels.shape[0]-ws)))
        #print(data.shape,traindata.shape)
        for i in range((int(traindata.shape[0]/self.step_size)-self.window_size)):
            traindata[i] = data[i*self.step_size:(i*self.step_size)+self.window_size].T
            trainlabel[i] = label[i*self.step_size:(i*self.step_size)+self.window_size].T
            trainlabelrelax[i] = relaxlabel[i*self.step_size:(i*self.step_size)+self.window_size].T
            #sensorClasses[i] = Slabels[i]

        testlabel = torch.tensor(self.testlabel).reshape((testdata.shape[0],2))
        #testlabel = torch.tensor(testlabel).reshape((testdata.shape[0],2))
        tsensorClasses = torch.tensor(self.tsensor_labels)
        testrelaxlabel = torch.clone(testlabel)
        for i in range(testrelaxlabel.shape[0]):
            if testlabel[i,0]==1:
                testrelaxlabel[i-relax:i+relax,0]=1
            if testlabel[i,1]==1:
                testrelaxlabel[i-relax:i+relax,1]=1

        testdatac = torch.zeros((int(testdata.shape[0]/self.step_size),no_columns,self.window_size))
        testlabelc = torch.zeros((int(testlabel.shape[0]/self.step_size),2,self.window_size))
        testrelaxlabelc = torch.zeros((int(testlabel.shape[0]/self.step_size),2,self.window_size))
        #tsensorClasses = torch.zeros((int(tSlabels.shape[0]-ws)))
        for i in range(testdata.shape[0]-ws):
            testdatac[i] = testdata[i:i+ws].T
            testlabelc[i] = testlabel[i:i+ws].T
            testrelaxlabelc[i] = testrelaxlabel[i:i+ws].T
            #tsensorClasses[i] = tSlabels[i]
            
        return data,traindata,label,relaxlabel,trainlabel,trainlabelrelax,sensorClasses,testdata,testdatac,testlabel,testlabelc,testrelaxlabel,testrelaxlabelc,tsensorClasses
        """if chunked:
            return traindata,label,testdata,testlabel
        
        if Sclass:
            return traindata,sensorClasses,testdatac,tsensorClasses
        if relaxlabels:
            return traindata,trainlabelrelax,testdatac,testrelaxlabelc
        else:
            return traindata,trainlabel,testdatac,testlabelc"""

            

def yoloLabeler(label):
    
    totalsteps = torch.where(label[:,0]==1)[0].shape[0]
    steps = label[:,0]+label[:,1]  #all steps in one column
    whe = torch.where(steps==1)[0] #indexes of all steps 
    midpoints = ((whe[1:]+whe[:-1])/2).int()[::2] # find the midpoint indexes of all steps. which is average of all values with jump of 1 
    lenth = ((whe[1:]-whe[:-1])/2).int()[::2] # find the length of all steps from midpoints.which is difference of all values with jump of 1
    table = {idx.item():ke.item() for idx,ke in zip(midpoints,lenth)} #create table of all midpoints and their consective length
    
    before = time.time()
    yoloLabels = np.zeros((label.shape[0],5,2,3))   #5,2,3 --- 5 grid cells, 2 bounding boxes, 3(x,w,c)
    startpoint=[]
    i=0
    for j in range(len(midpoints)):
        if j%1000==0:
            print(i,j,(time.time()-before)/60) 

        while i<int(label.shape[0]): 
            window_index = [i,i+200]   #start and end index of window
            grid_indexes = [i,i+40,i+80,i+120,i+160,i+200]  # start indexes of each grid, last one is for if conditions range.
            if midpoints[j]>window_index[0] and midpoints[j]<window_index[1]:   #check if mid of step is in this window
                for l in range(5):   # check for each grid 
                    if midpoints[j]>=grid_indexes[l] and midpoints[j]<grid_indexes[l+1]:     #check if it belong to this grid
                        if yoloLabels[i,l,0,2]==0: #if one bbx is already used then use second one
                            yoloLabels[i,l,0,0] = abs(grid_indexes[l]-midpoints[j])    #x --distance from start of the grid
                            yoloLabels[i,l,0,1] = table[midpoints[j].item()]   #width
                            yoloLabels[i,l,0,2] = 1 #confidence
                        else:
                            yoloLabels[i,l,1,0] = abs(grid_indexes[l]-midpoints[j])    #x
                            yoloLabels[i,l,1,1] = table[midpoints[j].item()]   #w
                            yoloLabels[i,l,1,2] = 1         #c               


            i=i+1
            if midpoints[j]<window_index[0]:   #if midpoint is no longer in the window range break the inner loop
                break
        i=midpoints[j].item()-500     # start from last midpoint minus 500 index. avoid looping from start
    print((time.time()-before)/60)
    
    return yoloLabels