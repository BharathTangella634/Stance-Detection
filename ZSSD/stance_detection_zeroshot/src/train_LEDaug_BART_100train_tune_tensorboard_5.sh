# nohup bash ./train_LEDaug_BART_100train_tune_tensorboard_5.sh ../config/config-bert.txt > train_LEDaug_BART_100train_tune_tensorboard_5_results.log 2>&1 &


# run the below bash code to run this shell file

# bash ./train_LEDaug_BART_100train_tune_tensorboard_5.sh ../config/config-bert.txt

###################################################################################################################
###################################################################################################################
###################################################################################################################
train_data=../data/raw_train_all_onecol.csv
dev_data=../data/raw_val_all_onecol.csv
test_data=../data/raw_test_all_onecol.csv





for seed in {0..3}
do
  echo "start random seed ${seed}......"
  
  echo "start training Gen ${epoch}......"
  python train_model.py -c $1 -train ${train_data} -dev ${dev_data} -test ${test_data} \
                        -s ${seed} -d 0.1 -clipgrad True -step 3  --earlystopping_step 5 -p 100
  
done


