import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_prob):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,dropout=0.2)
        #self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_size, 400)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0,c0))
        #out = self.dropout(out)
        out = out[:, -1, :]
        out = self.fc(out)
        #out = self.sigmoid(out)

        return out

class CNN2h(nn.Module):
    def __init__(self,ws,chanel,input_shape):
        super(CNN2h,self).__init__()
        self.ws =ws
        
        self.block1 = nn.Sequential(nn.Conv1d(chanel,16,kernel_size=5, padding=1 ),
                                   nn.MaxPool1d(kernel_size=3, stride=2),
                                   nn.BatchNorm1d(16),
                                   nn.ReLU())

        self.block2 = nn.Sequential(nn.Conv1d(16,32,kernel_size=5, padding=1 ),
                                   nn.MaxPool1d(kernel_size=3, stride=2),
                                   nn.BatchNorm1d(32),
                                   nn.ReLU())

        
        self.out_shape = self.block2(self.block1(torch.ones(input_shape))).flatten(start_dim=1)

        
        self.mlpS = nn.Sequential(nn.Linear(self.out_shape.shape[1],256),
                                 nn.ReLU(),
                                 nn.Dropout(0.5),
                                 #nn.Linear(256,128),
                                 #nn.ReLU(),
                                 #nn.Dropout(0.5),
                                 nn.Linear(256,self.ws))

        self.mlpE = nn.Sequential(nn.Linear(self.out_shape.shape[1],256),
                                 nn.ReLU(),
                                 nn.Dropout(0.5),
                                 #nn.Linear(256,128),
                                 #nn.ReLU(),
                                 #nn.Dropout(0.5),
                                 nn.Linear(256,self.ws))
       
    def forward(self, inpu):
        out = self.block1(inpu)        
        out = self.block2(out)
        out1 = self.mlpS(out.flatten(start_dim=1))
        out2 = self.mlpE(out.flatten(start_dim=1))
        return out1,out2   


class CNN_class(nn.Module):
    def __init__(self,num_classes,chanel,input_shape):
        super(CNN_class,self).__init__()
        self.num_classes =num_classes
        
        self.block1 = nn.Sequential(nn.Conv1d(chanel,16,kernel_size=3, padding=1 ),
                                   nn.MaxPool1d(kernel_size=3, stride=1),
                                   nn.BatchNorm1d(16),
                                   nn.ReLU())

        self.block2 = nn.Sequential(nn.Conv1d(16,32,kernel_size=3, padding=1 ),
                                   nn.MaxPool1d(kernel_size=3, stride=1),
                                   nn.BatchNorm1d(32),
                                   nn.ReLU())

        
        self.out_shape = self.block2(self.block1(torch.ones(input_shape))).flatten(start_dim=1)
        #self.out_shape = self.block1(torch.ones(input_shape)).flatten(start_dim=1)
        
        self.mlp = nn.Sequential(nn.Linear(self.out_shape.shape[1],256),
                                 nn.ReLU(),
                                 nn.Dropout(0.5),
                                 nn.Linear(256,128),
                                 nn.ReLU(),
                                 nn.Dropout(0.5),
                                 nn.Linear(128,self.num_classes))
       
    def forward(self, inpu):
        out = self.block1(inpu)        
        out = self.block2(out)
        out = self.mlp(out.flatten(start_dim=1))
        return out  


class yolo(nn.Module):
    #input---->5,2,3-->5,6 [x1,w1,c1],[x2,w2,c2].......[x00,w00,c00][x10,w10,c10],[x01,w01,c01][x11,w11,c11]
    #output--->Batch,5,6 ---> 5 for each grid, 6--->[c1,c2,x1,x2,w1,w2]
    #output---> [x00,x01,x02,x03,x04,x10,x11,x12,x13,x14][w00,w01,w02,w03,w04,w10,w11,w12,w13,w14][c00,c01,c02,c03,c04,c10,c11,c12,c13,c14]
    #compare all x with all x in target 
    def __init__(self,ws,chanel,input_shape):
        super(yolo,self).__init__()
        self.ws =ws
        
        self.block1 = nn.Sequential(nn.Conv1d(chanel,128,kernel_size=3, padding=1 ),
                                   nn.MaxPool1d(kernel_size=3, stride=2),
                                   nn.BatchNorm1d(128),
                                   nn.ReLU())

        self.block2 = nn.Sequential(nn.Conv1d(128,64,kernel_size=3, padding=1 ),
                                   nn.MaxPool1d(kernel_size=3, stride=2),
                                   nn.BatchNorm1d(64),
                                   nn.ReLU())
        self.block3 = nn.Sequential(nn.Conv1d(64,16,kernel_size=3, padding=1 ),
                                   nn.MaxPool1d(kernel_size=3, stride=2),
                                   nn.BatchNorm1d(16),
                                   nn.ReLU())
        self.block4 = nn.Sequential(nn.Conv1d(16,5,kernel_size=3, padding=1 ),
                                   nn.MaxPool1d(kernel_size=3, stride=2))
        
        self.size = self.block2(self.block1(torch.zeros(input_shape))).flatten(start_dim=1).shape
        print(self.size)
        self.block5 = nn.Sequential(nn.Linear(self.size[1],1024),
                                    nn.Dropout(0.3),
                                   nn.ReLU(),
                                   nn.Linear(1024,30))
        self.c = nn.Sequential(nn.Linear(30,10),
                                       nn.ReLU(),
                                       nn.Linear(10,10))
        self.x = nn.Sequential(nn.Linear(30,10),
                                       nn.ReLU(),
                                       nn.Linear(10,10))
        self.w = nn.Sequential(nn.Linear(30,10),
                                       nn.ReLU(),
                                       nn.Linear(10,10))
    def forward(self, inpu):
        out = self.block1(inpu)        
        out = self.block2(out)
        #out = self.block3(out)
        #out = self.block4(out)
        out = self.block5(out.flatten(start_dim=1))
        out_c = self.c(out)
        out_w = self.w(out)
        out_x = self.x(out)
        out = torch.stack((out_x,out_w,out_c),axis=-1)
        return out

    
