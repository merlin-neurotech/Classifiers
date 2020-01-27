# Blink Classifiers

## Model 1

### Data

This model is trained with the following involuntary eye blinks dataset:

>Mohit Agarwal, Raghupathy Sivakumar
>BLINK: A Fully Automated Unsupervised Algorithm for Eye-Blink Detection in EEG Signals
>2019 56th Annual Allerton Conference on Communication, Control, and Computing (Allerton). IEEE, 2019


    Sampling Frequency: 250.0 samples per second
    Electrodes: Fp1 and Fp2
    Total Subjects: 20
    Experiment: Subjects were asked to blink when an external stimulation was provided



### Filtering

Each EEG run is filtered with the following parameters `ftype='butter', band='bandpass', frequency=(4,40), order=8, sampling_rate=250` using `biosppy`'s signals module (which in turn uses `scipy`).



### Epoching and Labelling

Each (now filtered) EEG run was epoched and labelled using the `epoch_and_label` function from the `preprocessing` module in this package. The following parameters were used `data=EEG_run, sampling_rate=250, timestamps, window_size=0.5, inter_window_interval=0.1`, where timestamps is an array with the timestamps of each blink. An epoch was labelled as a blink if a blink occured at any time during it.

The epochs of all the EEG runs were then combined into a single labelled dataset.



### Train-Validation-Test Split

The above preprocessing resulted in `20139` labelled examples, 5% of which (`1007`) were set aside as an independent testing set. The rest was used for training and validation with 20% being for validation.



### Model Architecture 

| Layer (type) | Output Shape | Param # |   
| --- | --- | --- |
| conv1d_1 (Conv1D) | (None, 116, 100) | 2100 |      
| conv1d_2 (Conv1D)            | (None, 107, 100) | 100100 |   
| max_pooling1d_1 (MaxPooling1D) | (None, 35, 100) | 0 |        
| conv1d_3 (Conv1D)            | (None, 26, 160) | 160160 |   
| conv1d_4 (Conv1D)            | (None, 17, 160) | 256160 |   
| global_average_pooling1d_1 (GlobalAveragePooling1D) | (None, 160) | 0 |        
| dropout_1 (Dropout)          | (None, 160) | 0 |        
| dense_1 (Dense)              | (None, 2) | 322 |      
| | | |
| Total params: | | 518,842 |



### Results and Evaluation

3 classifiers in total were trained using this architecture with different precision-recall tradeoffs. This is done in an attempt to cover as many usecases/applications as possible.

The most balanced model achieved the following results on the training set:

|  | precision | recall | f1-score | support | 
| --- | --- | --- | --- | --- |
| No-blink | 0.99 | 0.98 | 0.99 | 16503 | 
| Blink | 0.91 | 0.95 | 0.93 | 2629 | 
| 
| micro avg | 0.98 | 0.98  | 0.98 | 19132 | 
| macro avg | 0.95 | 0.97 | 0.96 | 19132 | 
| weighted avg | 0.98 | 0.98 | 0.98 | 19132 | 
| samples avg | 0.98| 0.98 | 0.98 | 19132 | 

And the following results on the testing set:

|  | precision | recall | f1-score | support | 
| --- | --- | --- | --- | --- |
| No-blink | 0.97 | 0.97 | 0.97 | 853 | 
| Blink | 0.82 | 0.82 | 0.82 | 154 | 
| 
| micro avg | 0.95 | 0.95  | 0.95 | 1007 | 
| macro avg | 0.90 | 0.90 | 0.90 | 1007 | 
| weighted avg | 0.95 | 0.95 | 0.95 | 1007 | 
| samples avg | 0.95 | 0.95 | 0.95 | 1007 |


The results for the other classifiers are similar but with a different precision-recall tradeoff. Note that the reported precision and recall in the file name of the classifiers is that of the blink class, which is smaller than that of the non-blink class, so any average of the metrics will be larger.   


## Using the Models in Your Project

To import the model into your project, use something like the following code:

```
from keras.models import load_model
model = load_model('../Models/model_1_balanced[.82 precision, .82 recall].h5')
```

and to make a prediction, use something like the following:

```
preds = (model.predict(epochs) > 0.5).astype(int)
```