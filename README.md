# CS492(I): Computer Vision Task


## Overview
This is the final repository for CS492(I) Computer Vision Task (supervised by Prof. Seunghoon Hong of KAIST CS) of Team 7. We have implemented our method, semisupervised learning on unlabeled data with label propagation and consistency regularization. 

### Members
* Juneseok Choi (20190665)
* Minyoung Hwang (20170738)
* Junghyun Lee(20170500)

## pretrain.py

### Options

### --name (default = 'nothing')

Defines the name of the session.

### --gpu_ids (default = '0')

Selects the ids of the GPUs to use.

### --batchsize (default varies depending on models)

Defines the batch size. If the value is set to -1, then the code chooses it to the maximum batch size possible for the selected model.

### --seed (default = 123)

The random seed.

### --model (default = 'resnet50')

Defines the model to use. ResNet18, ResNet50, ResNet101, ResNet152, Residual Attention Network 56, Residual Attention Network 92 are selected respectively for 'resnet18', 'resnet50', 'resnet101', 'resnet152', 'ran56', 'ran92'. The default batch sizes for single precision are 512, 256, 180, 128, 140, 80 in order. The default batch sizes for half precision are 1024, 512, 360, 256, 140, 80 in order. 

### --half (default = False)

If set to True, uses half precision except for the batch normalization layers in the model and optimization. 

### --optimizer (default = 'Adam')

Chooses the optimizer. Can be set to either 'Adam' (Adam optimizer) or 'SGD' (SGD optimizer)

### --LARS (default = False)

If set to True, uses LARS optimizer on top of the selected optimizer. 

### --adaptive_lr (default = False)

If set to True, automatically decides the learning rate based on the batch size (0.3 * batchsize / 256)

### scheduler (default = 'exp')

Chooses the scheduler. Can be set to 'linear' to choose MultiStepLR or to 'exp' to choose ExponentialLR.

### --lr (default varies depending on the model)

The learning rate. 

### --imResize (default = 256)

The image is resized to this value and then becomes imsize through RandomResizedCrop.

### --imsize (default = 224)

The image is resized to this value before being fed into the model.



## Options

The usual NSML commands are still applicable, and so we will only introduce the new options.
You can run it without any commands, then it will conduct a method we have proposed.
Notice that you should set at least one of them as true.

### --isUDA (default=True)

If true, then the model uses consistency loss term.

### --isLP (default=True)

if true, then the model uses label propagation.


## Dependencies

Since the running environment was NSML, basic packages such as torch were already installed. Thus we only write here packages that have to be newly installed into NSML.

* torchvision
* torchlars
* scipy
* numpy
* matplotlib
* pandas
* faiss-gpu


## Directories

### Best Model

Captured model with the best performance.

### codes

Stores all the codes.

### report

Stores LaTex raw files and the image files for the final report.
