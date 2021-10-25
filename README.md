# Automatic-Speech-Recognition
Repository for ASR course project: Comparing deep learning architectures for small-footprint keyword spotting

## Running the code

Install dependencies as:

```
pip install -r requirements.txt
```

To train a model, if you do not have preprocessed data, you can run this code from command line as:

```
python train.py --model=resnet_334k --preprocess=False
```

If you already have preprocessed data, set ```--preprocess=True```. If ```model``` parameter is not mentioned, default would be used.

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
|    └───data_info                                        // Metadata
|    |    testing_list.txt  
|    |    validation_list.txt                                 
|    |    ...   
|    └───logs                                             // Tensorboard logs
|    |      └───Cnn1Param99k 
|    |      └───ResNetParam334k                                                 
|    |      └───...
|    └───models                                           // Model save path
|    |      └───Cnn1Param99k 
|    |      └───ResNetParam334k                                                 
|    |      └───...
|    └───features                                         // Preprocessed features
|    |    train.npy
|    |    val.npy
|    |    ...
|    └───speech_data                                      // Speech data
|    |      └───backward
|    |      |    0a2b400e_nohash_0.wav
|    |      |    0a2b400e_nohash_1.wav
|    |      |    ...
|    |      └───bed
|    |      |    00f0204f_nohash_0.wav
|    |      |    00f0204f_nohash_0.wav
|    |      |    ...
|    |      └───cat
└───plots                                                 // Confusion matrix plots
|    plot_Cnn1Param99k.png
|    plot_Cnn1Param134k.png
|    ...
```

## Results

| Model | Precision | Recall | F1 Score | Accuracy |
|-------|-----------|--------|----------|----------|
| CNN1_99K | 0.6 | 0.59 | 0.59 | 0.59 |
| CNN1_100K | 0.68 | 0.68 | 0.68 | 0.68 |
| CNN1_134K | 0.7 | 0.69 | 0.69 | 0.69 |
| CNN2_248kK | 0.7 | 0.7 | 0.69 | 0.7 |
| CNN_BiLSTM_163K | 0.64 | 0.64 | 0.64 | 0.64 |
| ResNet_334K | 0.69 | 0.69 | 0.68 | 0.69 |
