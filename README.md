# Audio-Visual Speech Recognition (AVSR) - AVRelScore, VCAFE
This repository contains the PyTorch implementation of the following papers:
> **Watch or Listen: Robust Audio-Visual Speech Recognition with Visual Corruption Modeling and Reliability Scoring (CVPR2023) - AVRelScore**<br>
> Joanna Hong\*, Minsu Kim\*, Jeongsoo Choi, and Yong Man Ro (\*Equal contribution)<br>
> \[[Paper](https://arxiv.org/abs/2303.08536)\] \[[Demo Video](https://github.com/joannahong/AV-RelScore/tree/main/demo_video)\] <br><br>
> **Visual Context-driven Audio Feature Enhancement for Robust End-to-End Audio-Visual Speech Recognition (Interspeech 2022) - VCAFE** <br>
> Joanna Hong\*, Minsu Kim\*, and Yong Man Ro (\*Equal contribution)
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

### Occlusion Data Download
For visual corruption modeling, download `coco_object.7z` from the [repository](https://github.com/kennyvoo/face-occlusion-generation). 

Unzip and put the files at
```
./occlusion_patch/object_image_sr
./occlusion_patch/object_mask_x4
```

### Babble Noise Download
For audio corruption modeling, download babble noise file from [here](https://kaistackr-my.sharepoint.com/:u:/g/personal/ms_k_kaist_ac_kr/EbMad_UoWI5MiJH-EVHoA8YBzd0wMM4C_RnELPbMlmyTTg?e=lbnfdE). 

put the file at
```
./src/data/babbleNoise_resample_16K.npy
```

### Pre-trained Frontends 
For initializing visual frontend and audio frontend, please download the pre-trained models from the [repository](https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks#model-zoo). (resnet18_dctcn_audio/resnet18_dctcn_video)

Put the .tar file at
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
--data_type LRS2 \
--landmark_path '/path_to/LRS2_landmarks' \
--save_path '/path_to/LRS2_processed' 
```
```shell
# For LRS3
python preprocessing.py \
--data_path '/path_to/LRS3' \
--data_type LRS3 \
--landmark_path '/path_to/LRS3_landmarks' \
--save_path '/path_to/LRS3_processed' 
```

## Training the Model
Basically, you can choice model architecture with the parameter `architecture`. <br>
There are three options for the `architecture`: `AVRelScore`, `VCAFE`, `Conformer`. <br>
To train the model, run following command:

```shell
# AVRelScore: Distributed training example using 2 GPUs on LRS2
torchrun --standalone --nnodes=1 --nproc_per_node=2 \
train.py \
--data_path '/path_to/LRS2_processed' \
--data_type LRS2 \
--split_file ./data/LRS2/0_600.txt \
--model_conf ./src/models/model.json \
--checkpoint_dir 'enter_the_path_to_save' \
--v_frontend_checkpoint ./checkpoints/frontend/lrw_resnet18_dctcn_video.pth.tar \
--a_frontend_checkpoint ./checkpoints/frontend/lrw_resnet18_dctcn_audio.pth.tar \
--wandb_project 'wandb_project_name' \
--batch_size 2 \
--update_frequency 1 \
--epochs 200 \
--eval_step 5000 \
--visual_corruption \
--architecture AVRelScore \
--distributed \
--gpu 0,1
```

```shell
# AVRelScore: Distributed training example using 2 GPUs on LRS2 (Lower torch version)
python -m torch.distributed.launch --nproc_per_node=4 \
train.py \
--data_path '/path_to/LRS2_processed' \
--data_type LRS2 \
--split_file ./data/LRS2/0_600.txt \
--model_conf ./src/models/model.json \
--checkpoint_dir 'enter_the_path_to_save' \
--v_frontend_checkpoint ./checkpoints/frontend/lrw_resnet18_dctcn_video.pth.tar \
--a_frontend_checkpoint ./checkpoints/frontend/lrw_resnet18_dctcn_audio.pth.tar \
--wandb_project 'wandb_project_name' \
--batch_size 2 \
--update_frequency 1 \
--epochs 200 \
--eval_step 5000 \
--visual_corruption \
--architecture AVRelScore \
--distributed \
--gpu 0,1
```

```shell
# VCAFE: 1 GPU training example on LRS3
python train.py \
--data_path '/path_to/LRS3_processed' \
--data_type LRS3 \
--split_file ./data/LRS3/0_600.txt \
--model_conf ./src/models/model.json \
--checkpoint_dir 'enter_the_path_to_save' \
--v_frontend_checkpoint ./checkpoints/frontend/lrw_resnet18_dctcn_video.pth.tar \
--a_frontend_checkpoint ./checkpoints/frontend/lrw_resnet18_dctcn_audio.pth.tar \
--wandb_project 'wandb_project_name' \
--batch_size 2 \
--update_frequency 1 \
--epochs 200 \
--eval_step 5000 \
--visual_corruption \
--architecture VCAFE \
--gpu 0
```

Descriptions of training parameters are as follows:
- `--data_path`: Preprocessed Dataset location (LRS2 or LRS3)
- `--data_type`: Choose to train on LRS2 or LRS3
- `--split_file`: train and validation file lists (we train with files having maximum 600 frames)
- `--checkpoint_dir`: directory for saving checkpoints
- `--checkpoint`: saved checkpoint where the training is resumed from
- `--model_conf`: model_configuration
- `--wandb_project`: if want to use wandb, please set the project name here. 
- `--batch_size`: batch size
- `--update_frequency`: update_frquency, if you use too small batch_size increase update_frequency. Training batch_size = batch_size * udpate_frequency
- `--epochs`: number of epochs
- `--tot_iters`: if set, the train is finished at the total iterations set
- `--eval_step`: every step for performing evaluation
- `--fast_validate`: if set, validation is performed for a subset of validation data
- `--visual_corruption`: if set, we apply visual corruption modeling during training
- `--architecture`: choose which architecture will be trained. (options: AVRelScore, VCAFE, Conformer)
- `--gpu`: gpu number for training
- `--distributed`: if set, distributed training is performed
- Refer to `train.py` for the other training parameters

### check the training logs
```shell
tensorboard --logdir='./runs/logs to watch' --host='ip address of the server'
```
The tensorboard shows the training and validation loss, evaluation metrics.
Also, if you set `wandb_project`, you can check wandb log.

## Testing the Model
To test the model, run following command:
```shell
# AVRelScore: test example on LRS2
python test.py \
--data_path '/path_to/LRS2_processed' \
--data_type LRS2\
--model_conf ./src/models/model.json \
--split_file ./src/data/LRS2/test.ref \
--checkpoint 'enter_the_checkpoint_path' \
--architecture AVRelScore \
--results_path './test_results.txt' \
--rnnlm ./checkpoints/LM/model.pth \
--rnnlm_conf ./checkpoints/LM/model.json \
--beam_size 40 \
--ctc_weight 0.1 \
--lm_weight 0.5 \
--gpu 0
```

Descriptions of testing parameters are as follows:
- `--data_path`: Preprocessed Dataset location (LRS2 or LRS3)
- `--data_type`: Choose to train on LRS2 or LRS3
- `--split_file`: set to test.ref (./src/data/LRS2./test.ref or ./src/data/LRS3/test.ref)
- `--checkpoint`: model for testing
- `--model_conf`: model_configuration
- `--architecture`: choose which architecture will be trained. (options: AVRelScore, VCAFE, Conformer)
- `--gpu`: gpu number for training
- `--rnnlm`: language model checkpoint
- `--rnnlm_conf`: language model configuration
- `--beam_size`: beam size
- `--ctc_weight`: ctc weight for joint decoding
- `--lm_weight`: language model weight for decoding
- Refer to `test.py` for the other parameters


## Pre-trained model checkpoints
We release the pre-trained AVSR models (VCAFE and AVRelScore) on LRS2 and LRS3 datasbases. (Below WERs can be obtained at `beam_width`: 40, `ctc_weight`: 0.1, `lm_weight`: 0.5) 

| Model |       Dataset       |   WER   |
|:-------------------:|:-------------------:|:--------:|
|VCAFE|LRS2 |   [4.459](https://kaistackr-my.sharepoint.com/:u:/g/personal/ms_k_kaist_ac_kr/EUU96MS_vEtLvvBsIfTPiu8B72AUDqHE855b5o-sc3imaQ?e=0RMvAy)  |
|VCAFE|LRS3 |   [2.821](https://kaistackr-my.sharepoint.com/:u:/g/personal/ms_k_kaist_ac_kr/Ef_kW5KR4qhFoPB2De2WdpMBWZDq40GxNwUWqb_o7xIV8Q?e=QHvQOQ)  |
|AVRelScore|LRS2 |   [4.129](https://kaistackr-my.sharepoint.com/:u:/g/personal/ms_k_kaist_ac_kr/EfNfnFcbIU1FjkTM5aoIF0MBBXv2qYD3pF4G1VHbor2kHw?e=FxIxPC)  |
|AVRelScore|LRS3 |   [2.770](https://kaistackr-my.sharepoint.com/:u:/g/personal/ms_k_kaist_ac_kr/Edm9bMf7IchHr39jBcLF_6EB3H4WpoLyrS3lHaGRQaJKXA?e=da4cxS)  |

You can find the pre-trained Language Model in the following [repository](https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages#Model-Zoo).
Put the language model at
```
./checkpoints/LM/model.pth
./checkpoints/LM/model.json
```

## Testing under Audio-Visual Noise Condition
Please refer to the following [repository](https://github.com/joannahong/AV-RelScore) for making the audio-visual corrupted dataset.

## Acknowledgment
The code are based on the following two repositories, [ESPNet](https://github.com/espnet/espnet) and [VSR for Multiple Languages](https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages).

## Citation
If you find this work useful in your research, please cite the papers:
```
@inproceedings{hong2023watch,
  title={Watch or Listen: Robust Audio-Visual Speech Recognition with Visual Corruption Modeling and Reliability Scoring},
  author={Hong, Joanna and Kim, Minsu and Choi, Jeongsoo and Ro, Yong Man},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={18783--18794},
  year={2023}
}
```
```
@inproceedings{hong2022visual,
  title={Visual Context-driven Audio Feature Enhancement for Robust End-to-End Audio-Visual Speech Recognition},
  author={Hong, Joanna and Kim, Minsu and Ro, Yong Man},
  booktitle={23rd Annual Conference of the International Speech Communication Association, INTERSPEECH 2022},
  pages={2838--2842},
  year={2022},
  organization={International Speech Communication Association}
}
```

