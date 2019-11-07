export PYTHONPATH=./:$PYTHONPATH
#model_dir=./models/inv_adp_label_smooth_pair_cos_cifar10/
model_dir=./models/inv_adp_label_smooth_pair_normL2_cifar10/
mkdir -p $model_dir
CUDA_VISIBLE_DEVICES=0 nohup python fs_main.py \
    --resume \
    --adv_mode='attack_inversion' \
    --lr=0.1 \
    --model_dir=$model_dir \
    --init_model_pass=latest \
    --max_epoch=200 \
    --save_epochs=100 \
    --decay_epoch1=60 \
    --decay_epoch2=90 \
    --batch_size_train=60 \
    --dataset=cifar10 > log_train_inv_adp_label_smooth_pair_normL2_cifar10.txt &

