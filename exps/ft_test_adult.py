#!/usr/bin/env bash

import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', type=str, default="0")
    parser.add_argument('--node', type=str, default="0015")
    opt = parser.parse_args()

    return opt
args = parse_args()

os.system(f"CUDA_VISIBLE_DEVICES=0,1,2,3 python test.py \
--gpu 0 \
-gen_bs 1024 \
-dis_bs 512 \
--dataset asia_table \
--bottom_width 8 \
--img_size 8 \
--max_iter 10000 \
--gen_model ft_transformer_gen \
--dis_model ft_transformer_dis \
--df_dim 384 \
--d_heads 8 \
--d_depth 3 \
--g_depth 5,4,2 \
--dropout 0.2 \
--latent_dim 256 \
--gf_dim 1024 \
--num_workers 16 \
--g_lr 0.0001 \
--d_lr 0.0001 \
--optimizer adam \
--loss standard \
--wd 1e-3 \
--beta1 0 \
--beta2 0.99 \
--phi 1 \
--eval_batch_size 1024 \
--num_eval_imgs 32768 \
--init_type xavier_uniform \
--n_critic 4 \
--val_freq 20 \
--print_freq 50 \
--grow_steps 0 0 \
--fade_in 0 \
--patch_size 2 \
--ema_kimg 500 \
--ema_warmup 0.1 \
--ema 0.9999 \
--load_path  /home/fjd5166/tabular/transgan/TransGAN/logs/ft_train_2022_03_07_20_21_27/Model/checkpoint \
--exp_name ft_test_adult")