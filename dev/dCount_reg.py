from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torchvision import transforms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pims
import pathlib
import torch.optim as optim
from torch.autograd import Variable
import torch
torch.cuda.empty_cache()
import skimage as skm
import glob
import sys

#classes/functions

class DefectDataset(Dataset):
    """custom dataset for star locator"""

    def __init__(self, label_dir, root_dir):
        """
        Args:
            csvfile (pathlib): folder containing labels
            root_dir (pathlib): root directory with all the images
        """
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.labels = self.process_labels()
        self.frames = pims.ImageSequence(str(root_dir / '*.tiff')) 
        self.shape = self.frames[1].shape
        ar = np.array(self.frames).reshape(-1,4)
        self.meand = ar.mean(axis=0)
        self.stdd = ar.std(axis=0)
        ar = None
        self.n = self.shape[0]
        #self.to_tensor = transforms.Compose([transforms.ToTensor(), transforms.Normalize(tuple(self.meand[0:3]),tuple(self.stdd[0:3]))])
        self.to_tensor = transforms.ToTensor()

    def process_labels(self):
        '''
        we need a function that will read in all the annotations and create a master dataframe [img_name][total_defect]
        '''
        files = glob.glob(str(self.label_dir / '*.dat'))
        df = pd.DataFrame([[f,np.count_nonzero(np.genfromtxt(f))] for f in files], columns = ['names', 'numbers'])
        df['frame'] = df['names'].str.extract('label-t_(\d*)\.dat').astype('int')
        df = df.sort_values('frame').reset_index(drop='True')
        #df.set_index('frame', inplace=True)
        return(df)

    def __len__(self):
        return len(self.frames)
 

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx.tolist()

        sample = [self.to_tensor(self.frames[idx][:,:,0:3]/255).float(), torch.tensor(self.labels.iloc[idx]['numbers']).float()/256**2]
        return sample
        

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        cLayers1 = 16
        cLayers2 = 8
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size = 3, padding = 1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 3, padding = 1)
        self.conv3 = nn.Conv2d(64, 1, kernel_size = 3, padding = 1)
        self.fc1 = nn.Linear(1 * 4*4, 1)

        self.fc3 = nn.Linear(1, 1, bias = True)

        self.pool = nn.MaxPool2d(4)
        self.act1 = nn.LeakyReLU()
        self.act1 = nn.ReLU()
        self.prob = nn.Sigmoid()
    def forward(self, x):
        out = self.pool( self.act1( self.conv1(x)))
        out = self.pool(self.act1(self.conv2(out)))
        out = self.pool(self.act1(self.conv3(out)))
        out = out.view(-1, 1*4*4) 
        out = self.act1(self.fc1(out))
        out = self.act1(self.fc3(out))

        return out
        #the output of this will be a 125*125 matrix.  

#first, read in data into dataloader, and split into test and validation set
data_folder = pathlib.Path('.') /'..' / 'data' /'img'
csv_name = pathlib.Path('.')/'..'/'data' / 'labels'

print(data_folder)
defectD = DefectDataset(csv_name, data_folder)
train_size = int(0.8*len(defectD))
test_size = len(defectD) - train_size

import datetime  # <1>
device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
print(f"Training on device {device}.")


def loss_funct(output, target):
    #loss = torch.sum((torch.mm((torch.exp( (output-target)**2)-1).t(),target))[:,0])
    #rel_error = torch.sum((output.T[0]-target)**2/(target+1)**2)
    error = torch.mean((output.sum(dim=1)-target)**2)
    return error #+.5*rel_error

def training_loop(n_epochs, optimizer, model, loss_fn, train_loader):
    for epoch in range(1, n_epochs + 1):  # <2>
        loss_train = 0.0
        for imgs, labels in train_loader:  # <3>
            imgs = imgs.to(device=device) 
            labels = labels.to(device=device)
            outputs = model(imgs)  # <4>
            
            
            loss = loss_fn(outputs, labels)  # <5>
            optimizer.zero_grad()  # <6>
            
            loss.backward()  # <7>
            
            optimizer.step()  # <8>

            loss_train += loss.item()  # <9>

        if epoch == 1 or epoch % 10 == 0:
            print('{} Epoch {}, Training loss {}, std{} , teststd {} '.format(
                datetime.datetime.now(), epoch, loss_train/len(train_loader), outputs.std(), labels.float().std()))

def validate(model, train_loader, val_loader):
    for name, loader in [('train', train_loader), ('val', val_loader)]:
        correct = 0
        total = 0

        with torch.no_grad():
            for imgs, labels in loader:
                imgs = imgs.to(device)
                labels = labels.to(device)
                outputs = model(imgs)
                total += labels.shape[0]
                cor = ((outputs - labels)*256**2).int()
                correct += int((cor == 0).sum())
        print("Accuracy {}: {:.2f}".format(name , correct / total))

def val2(model, train_loader, val_loader):
    for name, loader in [('train', train_loader), ('val', val_loader)]:
        ave_dist= 0
        total = 0
        right_counts = 0

        with torch.no_grad():
            for imgs, labels in loader:
                imgs = imgs.to(device)
                labels = labels.to(device)
                outputs = model(imgs)
                total += labels.shape[0]
                num= outputs.argmax(dim=1)
                cor =  (num-labels)
                right_counts += int((cor == 0).sum())
                ave_dist = torch.sqrt( ((num-labels)**2).float().sum())
        print("Right percent {}, dist to truth {}".format(right_counts/total , ave_dist))


train_dataset, test_dataset = torch.utils.data.random_split(defectD, [train_size, test_size])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size =32 , shuffle = True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size =32, shuffle = False)
model = Net().to(device = device)
lr = 1e-4
optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay = .0005, momentum = .9)
loss_fn = loss_funct
loss_fn = nn.CrossEntropyLoss()
loss_fn = nn.MSELoss()
weight_file = pathlib.Path('./m1.pt')
if weight_file.is_file():
    loaded_model = Net().to(device)
    loaded_model.load_state_dict(torch.load(str(weight_file), map_location=torch.device(device)))
    val2(loaded_model, train_loader, test_loader)
else:
    run_more = lr
    while(run_more != 'n'):
        try:
            optimizer = optim.SGD(model.parameters(), lr=float(lr))
            training_loop(  # <5>
                n_epochs = 1000,
                optimizer = optimizer,
                model = model,
                loss_fn = loss_fn,
                train_loader = train_loader,
            )
        except KeyboardInterrupt:
            run_more = 'n'

    weight_file = pathlib.Path('./m1.pt')
    torch.save(model.state_dict(), str(weight_file)) 

    val2(model, train_loader, test_loader)


def plt_frame(frame):
   f=test_dataset.__getitem__(frame)
   fig,ax = plt.subplots()
   ax.imshow(f[0].permute(1,2,0))
   outputs = model(f[0].unsqueeze(0).to(device))
   prob = torch.exp(outputs)/torch.exp(outputs).sum()
   thresh = .0015
   pred = torch.where(prob > thresh)
   ax.scatter(20,20, color = 'blue', s = 10)
   for i in pred:
       x = (i//31)*8
       y = (i % 31)*8
       print(x,y)
       ax.scatter(x.cpu(),y.cpu(),color = 'red', s = 10)

#plt_frame(10)
plt.show()
