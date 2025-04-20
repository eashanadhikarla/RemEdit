#!/bin/bash

sh_file_name="test.sh"
gpu="0"
# config="custom.yml"
config="celeba.yml"
guid="smiling"
test_step=40    # if large, it takes long time.
dt_lambda=1.0   # hyperparameter for dt_lambda. This is the method that will appear in the next paper.
CUDA_VISIBLE_DEVICES=$gpu
degree=0.9

python main.py  --run_test                              \
                --config $config                        \
                --exp ./runs/exp6/${guid}               \
                --edit_attr $guid                       \
                --do_train 0                            \
                --do_test 1                             \
                --n_train_img 5                         \
                --n_test_img 10                         \
                --bs_train 1                            \
                --n_inv_step 50                         \
                --n_train_step 40                       \
                --n_test_step $test_step                \
                --get_h_num 1                           \
                --train_delta_block                     \
                --sh_file_name $sh_file_name            \
                --save_x0                               \
                --use_x0_tensor                         \
                --hs_coeff_delta_h $degree              \
                --dt_lambda $dt_lambda                  \
                --add_noise_from_xt                     \
                --lpips_addnoise_th 1.2                 \
                --lpips_edit_th 0.9                     \
                --save_process_origin                   \
                --save_x_origin                         \
                --manual_checkpoint_name "/home/ubuntu/controlbfr/RiemannianEdit/src/checkpoint/smiling_LC_CelebA_HQ_t999_ninv50_ngen50_5.pth"

                # --lpips_addnoise_th 1.2
                # --lpips_edit_th 0.33

# --manual_checkpoint_name "smiling_LC_CelebA_HQ_t999_ninv40_ngen40_0.pth"    \