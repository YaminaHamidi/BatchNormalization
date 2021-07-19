import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def train(model, n_epochs, lr, trainLoader,testLoader):
    """train model and return loss and accuracy"""
    
    net = model
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr= lr)

    train_loss_overTime = []
    test_loss_overTime = []
    test_class_correct = list(0. for i in range(10))
    test_class_total = list(0. for i in range(10))
    train_class_correct = list(0. for i in range(10))
    train_class_total = list(0. for i in range(10))

    for epoch in range(n_epochs): 

            net.train()
            running_loss = 0.0
            n_batches = 0 

            for batch_i, data in enumerate(trainLoader):

                ## get the input images and their corresponding labels
                inputs, labels = data
                #print(len(labels))
                #print(inputs.size())
                
                ## zero the parameter (weight) gradients
                optimizer.zero_grad()

                ## forward pass to get outputs
                outputs = net(inputs)

                ## calculate the loss
                loss = criterion(outputs, labels)

                ## backward pass to calculate the parameters gradients
                loss.backward()

                ## update the parameters
                optimizer.step()

                ## extract loss
                running_loss += loss.item()
                n_batches+=1

                ## get the predicted class from the maximum value in the output-list of class scores
                _, predicted = torch.max(outputs.data, 1)
                
                ## number of correctly classified images in a batch
                correct = np.squeeze(predicted.eq(labels.data.view_as(predicted)))

                ## calculate test accuracy for each class
                for i in range(len(labels)):
                    label = labels.data[i]
                    train_class_correct[label] += correct[i].item()
                    train_class_total[label] += 1

            avg_loss   = running_loss / n_batches 
            train_loss_overTime.append(avg_loss)
            print('Epoch: {}, Avg. Train Loss: {}'.format(epoch + 1, avg_loss))

            net.eval()  
            running_loss = 0.0
            n_batches = 0 
            for batch_i, data in enumerate(testLoader):
                ## get the input images and their corresponding labels
                inputs, labels = data

                ## forward pass to get outputs
                outputs = net(inputs)

                ## get the predicted class from the maximum value in the output-list of class scores
                _, predicted = torch.max(outputs.data, 1)

                ## extract loss
                loss = criterion(outputs, labels) 
                running_loss += loss.item()
                n_batches+=1

                ## number of correctly classified images in a batch
                correct = np.squeeze(predicted.eq(labels.data.view_as(predicted)))

                ## calculate test accuracy for each class
                for i in range( len(labels) ):  #batch_size
                    label = labels.data[i]
                    test_class_correct[label] += correct[i].item()
                    test_class_total[label] += 1

            avg_loss   = running_loss / n_batches 
            test_loss_overTime.append(avg_loss)
            print('Epoch: {}, Avg. Test Loss: {}'.format(epoch + 1, avg_loss))   
    return net,train_loss_overTime, test_loss_overTime, train_class_correct, test_class_correct, train_class_total, test_class_total
                    
                    
def print_train_class_accuracy(net,train_class_correct,train_class_total):
    for i in range(10):
        print('Train Accuracy of %2d: %2d%% (%2d/%2d)' % (
            i+1, 100 * train_class_correct[i] / train_class_total[i],
            np.sum(train_class_correct[i]), np.sum(train_class_total[i])))
        
    print('\nTrain Accuracy (Overall): %2d%% (%2d/%2d)' % (
    100. * np.sum(train_class_correct) / np.sum(train_class_total),
    np.sum(train_class_correct), np.sum(train_class_total))) 

    print('\nTrain Error Rate', 1 - np.sum(train_class_correct) / np.sum(train_class_total) )
            
        
def print_test_class_accuracy(net,test_class_correct,test_class_total):
    for i in range(10):
        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
            i+1, 100 * test_class_correct[i] / test_class_total[i],
            np.sum(test_class_correct[i]), np.sum(test_class_total[i])))
        
    print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
        100. * np.sum(test_class_correct) / np.sum(test_class_total),
        np.sum(test_class_correct), np.sum(test_class_total)))        

    print('\nTest Error Rate', 1 - np.sum(test_class_correct) / np.sum(test_class_total) )
    
    
def plot_loss(train_loss_overTime, test_loss_overTime):
    plt.plot(train_loss_overTime,c='r',label="Train Loss")
    plt.plot(test_loss_overTime,c='b',label="Test Loss")
    plt.legend()
    plt.show()
    
    
def plot_classif_samples(net, testloader):
    
    ## number of samples to plot
    N = 20
    ## obtain one batch of test images
    dataiter = iter(testloader)
    images, labels = dataiter.next()
    ## get corresponding predictions
    preds = np.squeeze(net(images).data.max(1, keepdim=True)[1].numpy())
    images = images.numpy()[:N]

    ## plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(25, 4))
    for idx in np.arange(N):
        ax = fig.add_subplot(2, N/2, idx+1, xticks=[], yticks=[])
        ax.imshow(np.squeeze(images[idx]), cmap='gray')
        ax.set_title("{} ({})".format(preds[idx], labels[idx]),
                     color=("green" if preds[idx]==labels[idx] else "red"))
        
        
        
        