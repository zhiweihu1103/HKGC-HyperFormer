export LOG_PATH=./logs/wd50k.out
export SAVE_DIR_NAME=wd50k
export DATASET=wd50k
export CUDA=1
export MOE_MODE=True
export ABLATION_MODE=dismult

export HIDDEN_SIZE=400
export CONV_KERNEL_WIDTH=20
export CONV_KERNEL_HEIGHT=20

export NUM_EXPORTS=64
export NUM_TOPS=2

export LABEL_SMOOTH=0.9

nohup python -u run.py \
   --task train \
   --epoch 100 \
   --batch_size 256 \
   --device cuda:$CUDA \
   --dataset $DATASET \
   --ent_neighbor_num 3 \
   --rel_neighbor_num 6 \
   --ent_qual_neighbor_num 2 \
   --use_interacte True \
   --kge_lr 6e-4 \
   --kge_label_smoothing $LABEL_SMOOTH \
   --num_hidden_layers 8 \
   --num_attention_heads 2 \
   --input_dropout_prob 0.7 \
   --context_dropout_prob 0.1 \
   --qual_dropout_prob 0.3 \
   --attention_dropout_prob 0.1 \
   --hidden_dropout_prob 0.1 \
   --entity_dropout_prob 0.3 \
   --residual_dropout_prob 0.0 \
   --hidden_size $HIDDEN_SIZE \
   --intermediate_size 2048 \
   --initializer_range 0.02 \
   --conv_input_dropout_prob 0.2 \
   --conv_hidden_dropout_prob 0.5 \
   --conv_feature_dropout_prob 0.5 \
   --conv_padding 0 \
   --conv_number_channel 96 \
   --conv_kernel_size 9 \
   --conv_kernel_width $CONV_KERNEL_WIDTH \
   --conv_kernel_height $CONV_KERNEL_HEIGHT \
   --conv_permution_size 1 \
   --num_workers 32 \
   --pin_memory True \
   --moe_num_expert $NUM_EXPORTS \
   --moe_top_k $NUM_TOPS \
   --moe_mode $MOE_MODE \
   --dataset_mode statement \
   --train_mode with_valid \
   --ablation_mode $ABLATION_MODE \
   --save_dir_name $SAVE_DIR_NAME \
   > $LOG_PATH 2>&1 &  