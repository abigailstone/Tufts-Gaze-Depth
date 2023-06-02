# Tufts Gaze Depth Dataset 

Official dataset repository for the Tufts Gaze Depth Datset, as presented in the paper *Gaze Depth Estimation for Eye-Tracking Systems* at SPIE DCS 2023. 

## Dataset 

Our gaze dataset includes over 75,000 gaze vectors and predicted eye models, collected at a distance range of 1.9m - 6.4m. Participants were asked to fixate on each of 12 targets, positioned at a variety of heights and angles around a room. 

The `data` directory contains .csv files that have been split into training (80%) and testing (20%) data.

## Model

Load the dataset and train the model by running: 
```
python train.py [--datapath PATH/TO/DATASET] [--epochs #EPOCHS][--savepath PATH/TO/SAVE/MODEL]
``` 

Evaluate a saved model by running: 
```
python eval.py [--model /PATH/TO/SAVED/MODEL][--datapath PATH/TO/DATASET]
```



## Citation 