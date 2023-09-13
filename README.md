# Audio-Visual Speech Recognition
PyTorch implementation of "Watch or Listen: Robust Audio-Visual Speech Recognition with Visual Corruption Modeling and Reliability Scoring" (CVPR2023) and "Visual Context-driven Audio Feature Enhancement for Robust End-to-End Audio-Visual Speech Recognition" (Interspeech 2022)

# Lip to Speech Synthesis in the Wild with Multi-task Learning

This repository contains the PyTorch implementation of the following papers:
> **Watch or Listen: Robust Audio-Visual Speech Recognition with Visual Corruption Modeling and Reliability Scoring (CVPR2023)**<br>
> Joanna Hong*, Minsu Kim*, Jeongsoo Choi, and Yong Man Ro (*Equal contribution)<br>
> \[[Paper](https://arxiv.org/abs/2303.08536)\] \[[Demo Video]([https://github.com/joannahong/Lip-to-Speech-Synthesis-in-the-Wild](https://github.com/joannahong/AV-RelScore/tree/main/demo_video))\] <br>
> **Visual Context-driven Audio Feature Enhancement for Robust End-to-End Audio-Visual Speech Recognition (Interspeech 2022)** <br>
> Joanna Hong*, Minsu Kim*, and Yong Man Ro (*Equal contribution)
> \[[Paper](https://arxiv.org/abs/2207.06020)\]

<div align="center"><img width="80%" src="img/IMG.png?raw=true" /></div>

## Requirements
- python 3.8
- pytorch 1.8 ~ 1.12
- torchvision
- torchaudio
- ffmpeg
- av
- tensorboard
- scikit-image
- opencv-python
- pillow
- librosa
- scipy
- albumentations

## Preparation
### Dataset Download
LRS2/LRS3 dataset can be downloaded from the below link.
- https://www.robots.ox.ac.uk/~vgg/data/lip_reading/

### Landmark Download
For data preprocessing, download the landmark of LRS2 and LRS3 from the [repository](https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages#Model-Zoo). 
(Landmarks for "VSR for multiple languages models")

### Pre-trained Frontends 
For initializing visual frontend and audio frontend, please download the pre-trained models from the [repository](https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks#model-zoo).

Put the .tar file to
```
./checkpoints/frontend/lrw_resnet18_dctcn_audio.pth.tar
./checkpoints/frontend/lrw_resnet18_dctcn_video.pth.tar
```

### Preprocessing
After download the dataset and landmark, we 1) align and crop the lip centered video, 2) extract audio, 3) obtain aligned landmark.
We suppose the data directory is constructed as
```
LRS2
├── main
|   ├── *
|   |   └── *.mp4
|   |   └── *.txt
├── pretrain
|   ├── *
|   |   └── *.mp4
|   |   └── *.txt
```

```
LRS3
├── trainval
|   ├── *
|   |   └── *.mp4
|   |   └── *.txt
├── pretrain
|   ├── *
|   |   └── *.mp4
|   |   └── *.txt
├── test
|   ├── *
|   |   └── *.mp4
|   |   └── *.txt
```

Run preprocessing with the following commands:
```shell
# For LRS2
python preprocessing.py \
--data_path '/path_to/LRS2' \
--data_type 'LRS2'
--landmark_path '/path_to/LRS2_landmarks' \
--save_path '/path_to/LRS2_processed' \
```
```shell
# For LRS3
python preprocessing.py \
--data_path '/path_to/LRS3' \
--data_type 'LRS3'
--landmark_path '/path_to/LRS3_landmarks' \
--save_path '/path_to/LRS3_processed' \
```

## Training the Model
`data_name` argument is used to choose which dataset will be used. (LRS2 or LRS3) <br>
To train the model, run following command:

```shell
# Data Parallel training example using 2 GPUs on LRS2
python train.py \
--data '/data_dir_as_like/LRS2-BBC' \
--data_name 'LRS2'
--checkpoint_dir 'enter_the_path_to_save' \
--visual_front_checkpoint 'enter_the_visual_front_checkpoint' \
--asr_checkpoint 'enter_pretrained_ASR' \
--batch_size 16 \
--epochs 200 \
--eval_step 3000 \
--dataparallel \
--gpu 0,1
```

```shell
# 1 GPU training example on LRS3
python train.py \
--data '/data_dir_as_like/LRS3-TED' \
--data_name 'LRS3'
--checkpoint_dir 'enter_the_path_to_save' \
--visual_front_checkpoint 'enter_the_visual_front_checkpoint' \
--asr_checkpoint 'enter_pretrained_ASR' \
--batch_size 8 \
--epochs 200 \
--eval_step 3000 \
--gpu 0
```

Descriptions of training parameters are as follows:
- `--data`: Dataset location (LRS2 or LRS3)
- `--data_name`: Choose to train on LRS2 or LRS3
- `--checkpoint_dir`: directory for saving checkpoints
- `--checkpoint` : saved checkpoint where the training is resumed from
- `--asr_checkpoint` : pretrained ASR checkpoint
- `--batch_size`: batch size 
- `--epochs`: number of epochs 
- `--dataparallel`: Use DataParallel
- `--gpu`: gpu number for training
- `--lr`: learning rate
- `--output_content_on`: when the output content supervision is turned on (reconstruction loss)
- Refer to `train.py` for the other training parameters

The evaluation during training is performed for a subset of the validation dataset due to the heavy time costs of waveform conversion (griffin-lim). <br>
In order to evaluate the entire performance of the trained model run the test code (refer to "Testing the Model" section).

### check the training logs
```shell
tensorboard --logdir='./runs/logs to watch' --host='ip address of the server'
```
The tensorboard shows the training and validation loss, evaluation metrics, generated mel-spectrogram, and audio


## Testing the Model
To test the model, run following command:
```shell
# test example on LRS2
python test.py \
--data 'data_directory_path' \
--data_name 'LRS2'
--checkpoint 'enter_the_checkpoint_path' \
--batch_size 20 \
--gpu 0
```

Descriptions of training parameters are as follows:
- `--data`: Dataset location (LRS2 or LRS3)
- `--data_name`: Choose to train on LRS2 or LRS3
- `--checkpoint` : saved checkpoint where the training is resumed from
- `--batch_size`: batch size 
- `--dataparallel`: Use DataParallel
- `--gpu`: gpu number for training
- Refer to `test.py` for the other parameters


## Pre-trained model checkpoints
The pre-trained ASR models for output-level content supervision and lip-to-speech synthesis models on LRS2 and LRS3 are available. <br>

| Model |       Dataset       |   STOI   |
|:-------------------:|:-------------------:|:--------:|
|ASR|LRS2 |   [Link](https://kaistackr-my.sharepoint.com/:u:/g/personal/ms_k_kaist_ac_kr/EYjyTk0Bxy9CqLVmshqVXWEBlZc2Tq_4JnC4ox1tQ7jXOA?e=s8rZMW)  |
|ASR|LRS3 |   [Link](https://kaistackr-my.sharepoint.com/:u:/g/personal/ms_k_kaist_ac_kr/EcPkEXJ9UgNInxbJX_eh5aYBoZDLnxMY8AAEDNEiyBEJjw?e=uytxOK)  |
|Lip2Speech|LRS2 |   [0.526](https://kaistackr-my.sharepoint.com/:u:/g/personal/ms_k_kaist_ac_kr/EWD7vxY4S7pPjNE8dUwSMJwBdgPFunw62HsDLIuUlWcKAQ?e=XYdHfn)  |
|Lip2Speech|LRS3 |   [0.497](https://kaistackr-my.sharepoint.com/:u:/g/personal/ms_k_kaist_ac_kr/Ea9mi0aKAa1Gu53jTKiQV0IB6x7s2rI1mG9hkgBdBCYWWg?e=SRcK6o)  |


## Citation
If you find this work useful in your research, please cite the paper:
```
@article{kim2023lip,
  title={Lip-to-Speech Synthesis in the Wild with Multi-task Learning},
  author={Kim, Minsu and Hong, Joanna and Ro, Yong Man},
  journal={arXiv preprint arXiv:2302.08841},
  year={2023}
}
```

