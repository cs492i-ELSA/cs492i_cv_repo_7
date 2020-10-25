# CS492(I): Computer Vision Task


## Overview
This is the final repository for CS492(I) Computer Vision Task (supervised by Prof. Seunghoon Hong of KAIST CS) of Team 7. We have implemented our method, semisupervised learning on unlabeled data with label propagation and consistency regularization. 

### Members
* Juneseok Choi (20190665)
* Minyoung Hwang (20170738)
* Junghyun Lee(20170500)



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
