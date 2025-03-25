#!/bin/bash

sh_file_name="script_inference.sh"
gpu="0"
# config="custom.yml"
config="celeba.yml"
guid="smiling"
test_step=500    # if large, it takes long time.
dt_lambda=1.0   # hyperparameter for dt_lambda. This is the method that will appear in the next paper.
CUDA_VISIBLE_DEVICES=$gpu

python main.py  --run_test                                                     \
                --config $config                                               \
                --exp ./runs/temp/${guid}                                      \
                --edit_attr $guid                                              \
                --do_train 0                                                   \
                --do_test 1                                                    \
                --n_train_img 2                                                \
                --n_test_img 1                                                 \
                --bs_train 1                                                   \
                --n_inv_step 50                                                \
                --n_train_step 50                                              \
                --n_test_step $test_step                                       \
                --get_h_num 1                                                  \
                --train_delta_block                                            \
                --sh_file_name $sh_file_name                                   \
                --save_x0                                                      \
                --use_x0_tensor                                                \
                --hs_coeff_delta_h 0.2                                         \
                --dt_lambda $dt_lambda                                         \
                --add_noise_from_xt                                            \
                --lpips_addnoise_th 1.2                                        \
                --lpips_edit_th 0.33                                           \
                --save_process_origin                                          \
                --save_x_origin                                                \
                --manual_checkpoint_name "/home/ubuntu/controlbfr/asyrp/src/checkpoint/smiling_LC_CelebA_HQ_t999_ninv40_ngen40_0.pth" # \
                # --custom_train_dataset_dir "/home/ubuntu/controlbfr/dataset/CELEBA_HQ_RAW" \
                # --custom_test_dataset_dir "/home/ubuntu/controlbfr/dataset/GEN_AI_LR/open_eyes_crop"  \
                # --user_defined_t_edit 515  \
                # --user_defined_t_addnoise 170 \

                # "checkpoint/conv_smiling_LC_CelebA_HQ_t999_ninv40_ngen40_4.pth" \

                # if you did not compute lpips, use it.
                # --user_defined_t_edit 500                                    \
                # --user_defined_t_addnoise 200                                \

# --manual_checkpoint_name "smiling_LC_CelebA_HQ_t999_ninv40_ngen40_0.pth"    \