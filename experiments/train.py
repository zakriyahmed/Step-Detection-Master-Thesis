import torch
import torch.nn as nn
import torch.optim as optim


def trainCNN2h(model,dataloader,ws0,we0,ws1,we1,epochs=10,lr=0.0001):
    his=[]
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion0 = nn.BCEWithLogitsLoss(reduction='none')
    criterion1 = nn.BCEWithLogitsLoss(reduction='none')
    model = model.train()
    # Training loop
    for epoch in range(epochs):
        total_loss = 0.0

        for batch_inputs, batch_targets in dataloader:
            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(batch_inputs)

            # Compute the loss
            loss_s = criterion0(outputs[0], batch_targets[:,0,:])
            loss_e = criterion1(outputs[1], batch_targets[:,1,:])

            w1 = torch.ones((batch_targets.shape[0],batch_targets.shape[2]))*ws0
            #print(w1.shape,batch_targets.shape)
            w1[[batch_targets[:,0,:]==1]]=ws1
            w2 = torch.ones((batch_targets.shape[0],batch_targets.shape[2]))*we0
            w2[[batch_targets[:,1,:]==1]]=we1
            loss= (torch.mean(loss_s*w1) + torch.mean(loss_e*w2))/2

            #loss = (loss_s+loss_e)/2
            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Track the total loss
            total_loss += loss.item()
            his.append(loss.item())
        # Calculate average loss for the epoch
        avg_loss = total_loss / len(dataloader)

        # Print epoch statistics
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f}")
        
    return model,his


def trainYolo(model,dataloader,epochs=10,lr=0.0001):
    his=[]
    hisc= []
    hisx = []
    hisw = []
    optimizer = optim.Adam(model.parameters(), lr=lr)
    classLossbb1 = nn.MSELoss(reduction='none')#BCEWithLogitsLoss()
    classLossbb2 = nn.MSELoss(reduction='none')
    indexLossbb1 = nn.MSELoss(reduction='none')
    indexLossbb2 = nn.MSELoss(reduction='none')
    lengthLossbb1 = nn.MSELoss(reduction='none')
    lengthLossbb2 = nn.MSELoss(reduction='none')
    model = model.train()#.cuda()
    # Training loop
    for epoch in range(epochs):
        total_loss = 0.0

        for batch_inputs, batch_targets in dataloader:
            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(batch_inputs)
            #print(batch_targets[:,:,:,2].shape,outputs[:,:,0:2].shape)
            # Compute the loss
            
            target_bb1 = batch_targets[:,:,0,:]
            target_bb2 = batch_targets[:,:,1,:]
            #print(target_bb1.shape,target_bb1.get_device())
            w_bb1 = torch.ones(target_bb1[:,:,2].shape)
            w_bb2 = torch.ones(target_bb2[:,:,2].shape)
            #print(w_bb1.shape,target_bb1[:,:,2].shape)
            w_bb1[target_bb1[:,:,2]==0] = 1
            w_bb1[target_bb1[:,:,2]==1] = 1
            w_bb2[target_bb2[:,:,2]==0] = 1
            w_bb2[target_bb2[:,:,2]==1] = 1
            
            
            
            index_loss_bb1 = indexLossbb1(outputs[:,0:5,0],target_bb1[:,:,0]/39)*w_bb1
            index_loss_bb2 = indexLossbb2(outputs[:,5:10,0],target_bb2[:,:,0]/39)*w_bb2
            loss_i = torch.mean(index_loss_bb1) + torch.mean(index_loss_bb2)
            
            #loss_i = indexLoss(outputs[:,:,2:4], batch_targets[:,:,:,0]/39)
            
            length_loss_bb1 = lengthLossbb1(outputs[:,0:5,1],target_bb1[:,:,1]/53)*w_bb1
            length_loss_bb2 = lengthLossbb2(outputs[:,5:10,1],target_bb2[:,:,1]/53)*w_bb2
            loss_l = torch.mean(length_loss_bb1) + torch.mean(length_loss_bb2)
            
            class_loss_bb1 = classLossbb1(outputs[:,0:5,2],target_bb1[:,:,2])#*w_bb1
            class_loss_bb2 = classLossbb2(outputs[:,5:10,2],target_bb2[:,:,2])#*w_bb2
            loss_c = torch.mean(class_loss_bb1) + torch.mean(class_loss_bb2)
            
            #loss_i = indexLoss(outputs[:,:,2:4], batch_targets[:,:,:,0]/39)
            #loss_l = lengthLoss(outputs[:,:,4:6], batch_targets[:,:,:,1]/53)

            loss = loss_c + loss_i + loss_l


            loss.backward()
            optimizer.step()

            # Track the total loss
            total_loss += loss.item()
            his.append(loss.item())
            hisc.append(loss_c.item())
            hisx.append(loss_i.item())
            hisw.append(loss_l.item())
        # Calculate average loss for the epoch
        avg_loss = total_loss / len(dataloader)

        # Print epoch statistics
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f}")
        
    return model,[his,hisc,hisx,hisw]

def trainCNNClassifier(model,dataloader,testdataloader,epochs=10,lr=0.0001,cuda=True):
    his=[]
    acc=[]
    optimizer = optim.Adam(model.parameters(), lr=lr)
    accuracy_metric = Accuracy(task="multiclass", num_classes=4).cuda()
    criterion = nn.CrossEntropyLoss()
    if cuda:
        model = model.train().cuda()
    else:
        model = model.train()
    # Training loop
    for epoch in range(epochs):
        total_loss = 0.0
        model.train()
        for batch_inputs, batch_targets in dataloader:
            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(batch_inputs)

            # Compute the loss
            loss = criterion(outputs, batch_targets)

            #loss = (loss_s+loss_e)/2
            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Track the total loss
            total_loss += loss.item()
            his.append(loss.item())
        # Calculate average loss for the epoch
        avg_loss = total_loss / len(dataloader)

        model.eval()  # Set the model to evaluation mode
        val_predictions = []
        val_true_labels = []

        with torch.no_grad():
            
            for val_inputs, val_labels in testdataloader:
                
                val_outputs = model(val_inputs)
                _, val_preds = torch.max(val_outputs, 1)
                val_predictions.extend(val_preds.cpu().numpy())
                val_true_labels.extend(val_labels.cpu().numpy())
        

        # Calculate validation accuracy using the accuracy metric
        val_accuracy = accuracy_metric(torch.tensor(val_predictions), torch.tensor(val_true_labels))
        acc.append(val_accuracy.item())

        print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

        
        
    return model,his,acc

def train_lstm(model,dataloader,ws0,we0,ws1,we1,ws,epochs=10,lr=0.0001):
    his=[]
    criterion0 = nn.BCEWithLogitsLoss(reduction='none')
    criterion1 = nn.BCEWithLogitsLoss(reduction='none')
    model = model.train().cuda()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # Training loop
    for epoch in range(epochs):
        total_loss = 0.0

        for batch_inputs, batch_targets in dataloader:
            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass


            #plt.plot(batch_inputs[0].detach().cpu())
            #plt.plot(batch_targets[0,:,0].detach().cpu())
            #break

            outputs = model(batch_inputs)

            loss_s = criterion0(outputs[:,0:ws], batch_targets[:,:,0])
            loss_e = criterion1(outputs[:,ws:ws+ws], batch_targets[:,:,1])

            w1 = torch.ones((batch_targets.shape[0:2]),device='cuda')*ws0
            w1[[batch_targets[:,:,0]==1]]=ws1
            w2 = torch.ones((batch_targets.shape[0:2]),device='cuda')*we0
            w2[[batch_targets[:,:,1]==1]]=we1
            loss= (torch.mean(loss_s*w1) + torch.mean(loss_e*w2))/2

            #loss = (loss_s+loss_e)/2
            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Track the total loss
            total_loss += loss.item()
            his.append(loss.item())

        #break
        # Calculate average loss for the epoch
        avg_loss = total_loss / len(dataloader)

        # Print epoch statistics
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f}")
    return model,his