import torch
import torch.nn as nn
import torch.nn.functional as F



class LeNet5(nn.Module):
    """ LeCun et al. 1998 """  
    
    def __init__(self):
        super(LeNet5, self).__init__()

        self.conv1 = nn.Conv2d(1, 6, 5, 1, 0)
        self.conv2 = nn.Conv2d(6, 16, 5, 1, 0)
        self.conv3 = nn.Conv2d(16, 120, 5, 1, 0)
        
        self.avgpool = nn.AvgPool2d(2, 2) 
        self.tanh = nn.Tanh()       
        
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10) 
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.tanh(x)
        x = self.avgpool(x)
        
        x = self.conv2(x)
        x = self.tanh(x)
        x = self.avgpool(x)
        
        x = self.conv3(x) 
        x = self.tanh(x)

        x = x.view(x.shape[0], -1)
        
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.fc2(x)
        
        return x


def batch_normalization(X, gamma, beta, running_mean, running_var, eps,t):

    ## Training Mode 
    if torch.is_grad_enabled():
        if len(X.shape) == 2:
            mean = torch.mean(X,dim=0)
            var = torch.mean((X-mean)**2, dim=0)
        elif len(X.shape) == 4:
            mean = torch.mean( torch.mean( torch.mean(X, dim=3, keepdim=True) , dim=2, keepdim=True ), dim=0, keepdim=True)
            var = torch.mean( torch.mean( torch.mean((X-mean)**2, dim=3, keepdim=True) , dim=2 , keepdim=True), dim=0, keepdim=True)
        ## Batch Normalization
        X_norm = (X-mean) / torch.sqrt(var+eps)
        ## Update moving average statistics 
        running_mean = t * running_mean + (1.0 - t) * mean
        running_var = t * running_var + (1.0 - t) * var 
         
    ## Prediction Mode 
    else:
        X_norm = (X - running_mean) / torch.sqrt(running_var + eps)
        
    ## Scale and shift       
    Y = gamma * X_norm + beta  
    
    return Y, running_mean.data, running_var.data


class Batch_Normalization(nn.Module):

    def __init__(self, n_features, n_dimensions):
        super().__init__()
        
        if n_dimensions == 2:
            s = (1, n_features)
        else:
            s = (1, n_features, 1, 1)
            
        ## Running mean and var
        self.running_mean = torch.zeros(s)
        self.running_var = torch.ones(s)
        
        ## Learnable parameters
        self.gamma = nn.Parameter(torch.ones(s))
        self.beta = nn.Parameter(torch.zeros(s))
        
    def forward(self, X):
        Y, self.running_mean, self.running_var = batch_normalization(
            X, self.gamma, self.beta, self.running_mean, self.running_var, eps=1e-5, t=0.9)
        return Y
    

class LeNet5_batchNorm(nn.Module):
    """ LeCun et al. 1998 """ 
    """ Ioffe and Szegedy 2015 """  
    
    def __init__(self):
        super(LeNet5_batchNorm, self).__init__()

        self.conv1 = nn.Conv2d(1, 6, 5, 1, 0)
        self.conv2 = nn.Conv2d(6, 16, 5, 1, 0)
        self.conv3 = nn.Conv2d(16, 120, 5, 1, 0)
        
        self.avgpool = nn.AvgPool2d(2, 2) 
        self.tanh = nn.Tanh()       
        
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10) 
        
        self.BN1  = Batch_Normalization(6, 4)
        self.BN2  = Batch_Normalization(16, 4)
        self.BN3  = Batch_Normalization(120, 4)
        self.BN4  = Batch_Normalization(84, 2)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.BN1(x) 
        x = self.tanh(x)
        x = self.avgpool(x)
        
        x = self.conv2(x)
        x = self.BN2(x) 
        x = self.tanh(x)
        x = self.avgpool(x)
        
        x = self.conv3(x)
        x = self.BN3(x) 
        x = self.tanh(x)

        x = x.view(x.shape[0], -1)
        
        x = self.fc1(x)
        x = self.BN4(x)
        x = self.tanh(x)
        x = self.fc2(x)
        
        return x

    
