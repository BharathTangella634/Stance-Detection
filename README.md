# ModernBERT for Zero-Shot Stance Detection

This is the repository for the paper: ***"ModernBERT for Zero-Shot Stance Detection"***

Our code is developed based on python 3.10, Torch 2.5.1, CUDA 12.4. Experiments are performed on a single NVIDIA RTX A4000 GPU.

To run TTS for ZSSD task:
```
cd ./ZSSD/stance_detection_zeroshot/src
```
For 10% training setting:
```
bash ./train_LEDaug_BART_10train_tune_tensorboard_5.sh ../config/config-bert.txt 
```
For 100% training setting:
```
bash ./train_LEDaug_BART_100train_tune_tensorboard_5.sh ../config/config-bert.txt
```
