# STKPS-Net
This repo is an official implementation of "STKPS-Net: Spatio-Temporal Key Patch Selection Network for Few Shot Anomalous Action Recognition".
# Training & Evaluation
Experimental Setup
Hardware: 4 GPUs (--num_gpus 4)
Setting: 5-way 10-shot
Memory Optimization: Due to the high memory cost of 10-shot settings, query_per_class is set to 2 during training and testing.
## 1、Baseline + spatial adaptive key patch selection + long-short feature map spatio-temporal relation + spatio-temporal refined loss
Script: run_kinetics_v12_multiheadandmatchloss.py
### HMDB51
CUDA_VISIBLE_DEVICES=0,1,2,3 python run_kinetics_v12_multiheadandmatchloss.py \
    --dataset hmdb \
    -c ./result_hmdb_adafocus_v12_multiheadandmatchloss_shot10 \
    -i 20004 -r --test_iters 20002 \
    --shot 10 --query_per_class 2 --num_gpus 4 --query_per_class_test 2
### Kinetics
CUDA_VISIBLE_DEVICES=0,1,2,3 python run_kinetics_v12_multiheadandmatchloss.py \
    --dataset kinetics \
    -c ./result_kinetics_adafocus_v12_multiheadandmatchloss_shot10 \
    -i 30004 --test_iters 30002 \
    --shot 10 --query_per_class 2 --num_gpus 4 --query_per_class_test 2
### UCF-Crime v2
CUDA_VISIBLE_DEVICES=0,1,2,3 python run_kinetics_v12_multiheadandmatchloss.py \
    --dataset ucfc \
    -c ./result_kinetics_adafocus_v12_multiheadandmatchloss_shot10 \
    -i 30004 -r --test_iters 30002 \
    --shot 10 --query_per_class 2 --num_gpus 4 --query_per_class_test 2
## 2、Baseline + spatial adaptive key patch selection
Script: run_hmdbv6_sp_recoord.py
### HMDB51
CUDA_VISIBLE_DEVICES=0,1,2,3 python run_hmdbv6_sp_recoord.py \
    --dataset hmdb \
    -c ./result_hmdb_adafocus_v6_shot10 \
    -i 20004 --test_iters 20002 \
    --shot 10 --query_per_class 2 --num_gpus 4 --query_per_class_test 2
### Kinetics
CUDA_VISIBLE_DEVICES=0,1,2,3 python run_hmdbv6_sp_recoord.py \
    --dataset kinetics \
    -c ./result_kinetics_adafocus_v6_shot10 \
    -i 30004 --test_iters 30002 \
    --shot 10 --query_per_class 2 --num_gpus 4 --query_per_class_test 2
### UCF-Crime v2
CUDA_VISIBLE_DEVICES=0,1,2,3 python run_hmdbv6_sp_recoord.py \
    --dataset ucfc \
    -c ./result_kinetics_adafocus_v6_shot10 \
    -i 30004 -r --test_iters 30002 \
    --shot 10 --query_per_class 2 --num_gpus 4 --query_per_class_test 2
## 3、Baseline + spatial adaptive key patch selection + long-short feature map spatio-temporal relation
Script: run_kinetics_v10.py
### HMDB51
CUDA_VISIBLE_DEVICES=0,1,2,3 python run_kinetics_v10.py \
    --dataset hmdb \
    -c ./result_hmdb_adafocusv10sprecoord_shot10 \
    -i 20004 --test_iters 20002 \
    --shot 10 --query_per_class 2 --num_gpus 4 --query_per_class_test 2
### Kinetics
CUDA_VISIBLE_DEVICES=0,1,2,3 python run_kinetics_v10.py \
    --dataset kinetics \
    -c ./result_kinetics_adafocusv10sprecoord_shot10 \
    -i 30004 --test_iters 30002 \
    --shot 10 --query_per_class 2 --num_gpus 4 --query_per_class_test 2
### UCF-Crime v2
CUDA_VISIBLE_DEVICES=0,1,2,3 python run_kinetics_v10.py \
    --dataset ucfc \
    -c ./result_kinetics_adafocusv10sprecoord_shot10 \
    -i 30004 -r --test_iters 30002 \
    --shot 10 --query_per_class 2 --num_gpus 4 --query_per_class_test 2
