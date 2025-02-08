# run the below bash code to run this shell file

# bash ./train_LEDaug_BART_10train_tune_tensorboard_5.sh ../config/config-bert.txt

#!/usr/bin/env bash
# train_data= "C:\Users\CSE RGUKT\Downloads\TTS\TTS\TTS_zeroshot\data\raw_train_all_subset10_onecol.csv"
# dev_data= "C:\Users\CSE RGUKT\Downloads\TTS\TTS\TTS_zeroshot\data\raw_val_all_onecol.csv"
# test_data= "C:\Users\CSE RGUKT\Downloads\TTS\TTS\TTS_zeroshot\data\raw_test_all_onecol.csv"

# kg_data= "C:\Users\CSE RGUKT\Downloads\TTS\TTS\TTS_zeroshot\data\raw_train_all_subset_kg_epoch_onecol.csv"

train_data=../data/raw_train_all_subset10_onecol.csv
dev_data=../data/raw_val_all_onecol.csv
test_data=../data/raw_test_all_onecol.csv


for seed in {0..3}
do
    echo "start random seed ${seed}......"
    
    echo "start training Gen ${epoch}......"
    python train_model.py -c $1 -train ${train_data} -dev ${dev_data} -test ${test_data} \
                        -s ${seed} -d 0.1 -clipgrad True -step 3  --earlystopping_step 5 -p 10
done
###################################################################################################################
###################################################################################################################
###################################################################################################################

# Bharath
# Manideep
# Sathvik
# Sriram