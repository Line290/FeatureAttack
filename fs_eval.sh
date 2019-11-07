export PYTHONPATH=./:$PYTHONPATH
model_dir=./models/inv_adp_label_smooth_pair_cos_cifar10/
CUDA_VISIBLE_DEVICES=3 python fs_eval.py \
    --model_dir=$model_dir \
    --init_model_pass=latest \
    --attack=True \
    --attack_method_list=natural-pgd-cw\
    --dataset=cifar10 \
    --batch_size_test=80 \
    --resume
