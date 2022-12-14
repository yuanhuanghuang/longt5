#!/bin/bash -l

export DATA_PATH=../drop_data/                            # OpenNMT-preprocessed data path
export OPENNMT_PATH=../src/OpenNMT-py/                         # OpenNMT path
export SPLIT=train                    # Number of exposure examples (1 or 100)
export VSPLIT=validation
export SAVE_PATH=${OPENNMT_PATH}/tf_checkpoints # Save path for checkpoints
export SAVE_NAME=quality_longt5_8192      # Checkpoint name
export LOG_PATH=${OPENNMT_PATH}/logs           # Log path
export PRED_PATH=${OPENNMT_PATH}/preds         # Predictions path
export SEED=1                                  # Random seed
export CUDA_VISIBLE_DEVICES=0                 # GPU machine number

mkdir $LOG_PATH
mkdir $PRED_PATH
mkdir $SAVE_PATH

for SEED in 1
do
    # training
    python -m pdb $OPENNMT_PATH/train.py -mode mc -train_data $DATA_PATH/${SPLIT}_source.txt -train_datat $DATA_PATH/${SPLIT}_target.txt \
        -valid_data $DATA_PATH/${VSPLIT}_source.txt -valid_datat $DATA_PATH/${VSPLIT}_target.txt \
        -train_steps 30000 -valid_steps 100 -save_checkpoint_steps 500 -early_stopping 10 \
        -batch_size 2 -accum_count 32 -learning_rate 1e-3 \
        -task_name 'mc' -architect longt5 -load_t5 -trim_size 8192 \
        -save_model $SAVE_PATH/${SAVE_NAME}/s$SEED -layers 2 -rnn_size 512 -word_vec_size 512 -transformer_ff 512 -heads 4  \
        -encoder_type transformer -decoder_type transformer -position_encoding \
        -max_generator_batches 2 -dropout 0.1 -batch_type sents -normalization sents \
        -optim adafactor -adam_beta2 0.998 -decay_method none -warmup_steps 1000 \
        -max_grad_norm 0 -param_init 0  -param_init_glorot \
        -label_smoothing 0.0 --early_stopping_criteria accuracy \
        -world_size 1 -gpu_ranks 0 -seed $SEED --log_file ${LOG_PATH}/${SAVE_NAME}_s${SEED}.log

    # evaluation
    for SPLIT in validation
    do
        python $OPENNMT_PATH/translate.py -model $SAVE_PATH/${SAVE_NAME}/s${SEED}_best.pt \
                                          -src $DATA_PATH/${SPLIT}_source.txt \
                                          -tgt $DATA_PATH/${SPLIT}_target.txt \
                                          -output ${PRED_PATH}/${SPLIT}_pred_${SAVE_NAME}_s${SEED}.txt \
                                          -replace_unk -shard_size 0 \
                                          -gpu 0 -batch_size 1 \
                                          -max_length 200 -beam_size 5

    done
#
done

