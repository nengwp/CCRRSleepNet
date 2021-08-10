# CCRRSleepNet #

Code for the model in the paper [**CCRRSleepNet: A Hybrid Relational Inductive Biases Network for Automatic Sleep Stage Classification on Raw Single-Channel EEG by Neng Wenpeng, Lu Jun, and Xu Lei**](https://www.mdpi.com/2076-3425/11/4/456#authors). 



This work has been accepted for publication in [Brain Sciences](https://www.mdpi.com/2076-3425/11/4/456#authors).


## Prepare dataset ##
We evaluated our CCRRSleepNet with [Sleep-EDF](https://physionet.org/pn4/sleep-edfx/) dataset.

For the [Sleep-EDF](https://physionet.org/pn4/sleep-edfx/) dataset, you can run the following scripts to download SC subjects.

    cd data
    chmod +x download_physionet.sh
    ./download_physionet.sh

Then run the following script to extract specified EEG channels and their corresponding sleep stages.

    python prepare_physionet.py --data_dir data --output_dir data/eeg_fpz_cz --select_ch 'EEG Fpz-Cz'
    python prepare_physionet.py --data_dir data --output_dir data/eeg_pz_oz --select_ch 'EEG Pz-Oz'


## Training a model ##
Run this script to train a DeepSleepNet model for the first fold of the 20-fold cross-validation.

    python train.py --data_dir data/eeg_fpz_cz --output_dir output --n_folds 20 --fold_idx 0 --pretrain_epochs 100 --finetune_epochs 200 --resume False

You need to train a CCRRSleepNet model for every fold (i.e., `fold_idx=0...19`) before you can evaluate the performance. You can use the following script to run batch training

    chmod +x batch_train.sh
    ./batch_train.sh data/eeg_fpz_cz/ output 20 0 19 0


## Scoring sleep stages ##
Run this script to determine the sleep stages for the withheld subject for each cross-validation fold.

    python predict.py --data_dir data/eeg_fpz_cz --model_dir output --output_dir output

The output will be stored in numpy files.


## Get a summary ##
Run this script to show a summary of the performance of our CCRRSleepNet compared with the state-of-the-art hand-engineering approaches. The performance metrics are overall accuracy, per-class F1-score, and macro F1-score.

    python summary.py --data_dir output



## Acknowledgement ##
We refer to part of the code of  [DeepSleepNet](https://github.com/akaraspt/deepsleepnet/)