#!/bin/bash

sh_file_name="train.sh"
gpu="0"

config="celeba.yml"
guid="smiling"
CUDA_VISIBLE_DEVICES=$gpu

python main.py  --run_train                         \
                --config $config                    \
                --exp ./runs/exp6/$guid             \
                --edit_attr $guid                   \
                --do_train 1                        \
                --do_test 1                         \
                --bs_train 1                        \
                --bs_test 1                         \
                --lr_training 0.5                   \
                --n_train_img 50                    \
                --accumulation_steps 1              \
                --n_test_img 32                     \
                --n_inv_step 50                     \
                --n_train_step 50                   \
                --n_test_step 40                    \
                --get_h_num 1                       \
                --train_delta_block                 \
                --sh_file_name $sh_file_name        \
                --n_iter 10                         \
                --save_x0                           \
                --use_x0_tensor                     \
                --save_x_origin                     \
                --clip_loss_w 1.0                   \
                --l1_loss_w 3.0                     \
                --user_defined_t_edit 500           \
                --user_defined_t_addnoise 200       \
                --retrain 1                         \
                --t_0 999                           \

                # --clip_loss_w 0.8                 \
                # --l1_loss_w 3.0                   \

                # --load_random_noise               \
                # --user_defined_t_edit 513         \
                # --user_defined_t_addnoise 167     \

                # --n_train_img 1000                \
                # --n_test_step 40