# Automatic-Speech-Recognition
Repository for ASR course project: Comparing deep learning architectures for small-footprint keyword spotting

## Data Set
Google Speech Commands data set v0.02 was used for the project. Data can be downloaded from here: http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz


## Repository structure
```
Automatic-Speech-Recognition
│   README.md
|   train.py                                              // Script to train the models
|
└───src
|    └───scripts                                          // Scripts for data loading, preprocessing, augmentation and batch generation
|    |    augmenter.py
|    |    data_load.py
|    |    ...
|    └───models                                           // Scripts for various models and inferencing
|    |    cnn1_99k.py
|    |    resnet_334k.py
|    |    ...
|    └───util                                             // Utility scripts
|    |    definitions.py
|    |    └───config
|    |    |    cnn1_99k.yaml
|    |    |    resnet_334k.yaml
│    |    |    ...
└───data                                                  // Speech Commands Data Set v0.02
      └───data_info                                       // Metadata
      |    testing_list.txt  
      |    validation_list.txt                                 
      |    ...   
      |
      └───logs                                            // Tensorboard logs
      |      └───Cnn1Param99k 
      |      └───ResNetParam334k                                                 
      |      └───...
      └───models                                          // Model save path
      |      └───Cnn1Param99k 
      |      └───ResNetParam334k                                                 
      |      └───...
      └───features                                        // Preprocessed features
      |    train.npy
      |    val.npy
      |    ...
      └───speech_data                                     // Speech data
      |      └───backward
      |      |    0a2b400e_nohash_0.wav
      |      |    0a2b400e_nohash_1.wav
      |      |    ...
      |      └───bed
      |      |    00f0204f_nohash_0.wav
      |      |    00f0204f_nohash_0.wav
      |      |    ...
      |      └───cat
      |      |    ...
      
```
