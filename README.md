# Practice For Tensorflow

Table of Contents
================
1. [Introduction](#Introduction)
2. [Modules](#Modules)
    - [Read Data](#read_data)
    - [DL Project Template](#dl_template)

## Introduction
The intention of this repository is to collect the best practice modules to use tensorflow and complete machine learning tasks, like how to deal with big data, create
operation graph,etc, and may also include how to imporve the efficiency of tensorflow  just like this [repo](https://github.com/vahidk/EffectiveTensorflow).

## Modules
### Read Data <a name="read_data"></a>
[Here](https://github.com/syw2014/Practice4Tensorflow/blob/master/modules/read_data/tfrecords.py) is a simple example to indicate how to create TFRecord with multi-thread, 
and prefecth mechanism parse Tensorflow.SequenceExample.The original version was came from [this](https://github.com/AIChallenger/AI_Challenger), and I made some changes
and extention.You can check [thensorflow queuing](http://adventuresinmachinelearning.com/introduction-tensorflow-queuing/) for more detail explanation.

### A Simple DL Project Template <a name="dl_template"></a>
As a beginner of Tensorflow you may confuse how to create a completed project with your own model [repo](https://github.com/syw2014/Practice4Tensorflow/blob/master/modules/project_template.py), 
I had wrote a simple workflow for creating a deep learning project, there are seven parts as bellow,
    - 1 Define hyperparameters, which are the parameters you should define or set before training model, like learning rate, number of train stpes/ epoch, unit dimensions.
    - 2 Prepare input data, load data from files and do some pre-process to generate train X and train y(label).
    - 3 Construct model, define placeholders, networks like CNN/RNN, weights and bias, computational logic.
    - 4 Define loss function, here you can use some classical loss like cross-entropy, MSE, Maximum-Margin, and you need define you own loss function according your task.
    - 5 Create optimizer, the simplest way you create your optimizer is just choose one optimize algorithm, but there are more things you should consider, that whether
        your task need to do gradient clip, change optimizer after several steps.
    - 6 Define the single train step
    - 7 Create Tensorflow session, and initialize all variables
    - 8 Train your model
Each parts of the project will contains more tricks you should consider, this will be discussed in future practice.
