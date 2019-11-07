export PYTHONPATH=./:$PYTHONPATH
#model_dir=./models/feature_scatter_cifar10/
model_dir=./models/inv_adp_label_smooth_pair_cos_cifar10/
CUDA_VISIBLE_DEVICES=3 python feature_attack_batch.py \
    --model_dir=$model_dir \
    --init_model_pass=latest \
    --dataset=cifar10 \
    --batch_size_test=100 \
    --resume
