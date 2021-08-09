# BatchNormalization
Batch Normalization Implementation - Tests on MNIST digits with LeNet 


Data is traditionally only scaled and/or normalized, as part of the data pre-processing step, before input to the first layer in a Neural Network. 
It's well known that relatively large weights can cascade through the network and cause the gradient to vanish. 
Large data values can cause instabilities in the network. The larger the value the more likely it is that the sigmoid's value is gonna be larger and that it's derivative is gonna tend to zero.

To elaviate this problem the paper [1] suggests to scale each mini-batch of data before input into the hidden layers, which is believed to keep the distribution of the data stable throughout the training process. 

[1] : IOFFE, Sergey et SZEGEDY, Christian. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In : International conference on machine learning. PMLR, 2015. p. 448-456.
