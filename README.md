## Feature Attack

### *Important
 90% codes copy from [FeatureScatter][FS_github] and [Madry PGD adv. training][Madry_github]
 
 
### My work
Created Feature Attack, and it's stronger than PGD attack or CW attack w.r.t [Feature Scatter][FS_paper] and [Adversarial Interpolation Training][Adv_inter_paper].

### Reference Model
Model trained on CIFAR10: [FS][FS_model] and [Adv_inter][Adv_inter_model]
### Evaluate
##### Feature Scatter
```bash
sh fs_eval_feature_attack.sh
```
##### Madry's
```bash
cd cifar10_challenge && python feature_attack_batch_tf.py
```

### Result

| attack type                   | clean | FGSM  | PGD20-2-8 | CW20-2-8  | FeatureAttack100 | adv_test_images |
|-------------------------------|-------|-------|-------|-------|------------------|------------------
| Feature Scatter               | 90.3  | 78.4  | 71.1  | 62.4  | 36.94            |
| Adv_inter                     | 90.5  | 78.1  | 74.4  | 69.5  | 37.64            |
| Madry                         | 87.25 |       | 45.87 |       | 46.37            |
| [Sensible adversarial learning][sensible] | 91.51 | 74.32 | 62.04 | 59.91 | 43.76|[sensible_adv_x][sensible_adv_x_link]
| [bilateral_AT][BAT] ([mosa_eps4][BAT_code_link])| 92.8  |       | 71.0(pgd100-2-8)|67.9(CW100-2-8)|32.28|[bilater_adv_x][bilater_adv_x_link]
| [GCE][GCE_paper]| |62.74% | [9.55%(MNIST_PGD40-0.01-0.2)][GCE_github] | | 0|


##### Introduction of adversarial test images
For CIFAR10 test data set
```python
eps = 8./255.
nat_X = ALL_CLEAN_TEST_IMAGES  # default order in PyTorch [0, 1]
adv_X_uint8 = torch.load('ADV_TEST_IMAGES_PATH')
adv_X = adv_X_uint8.type(torch.FloatTensor) / 255.  # [0, 1]
assert adv_X.min() >= 0. and adv_X.max() <= 1.
abs_diff = torch.abs(adv_X - nat_X)
assert abs_diff <= eps + 0.0001
```



[FS_github]:https://github.com/Haichao-Zhang/FeatureScatter 
[Madry_github]:https://github.com/MadryLab/cifar10_challenge 
[FS_paper]:https://arxiv.org/pdf/1907.10764.pdf  
[Adv_inter_paper]:https://openreview.net/pdf?id=Syejj0NYvr  
[FS_model]:https://drive.google.com/open?id=1TCw1uVrAikOZIObHfALE-FuXXa7UKDDo  
[Adv_inter_model]:https://drive.google.com/open?id=1ak-Qovkra3oIqukAWc32rLJcAPkdpN79
[sensible]:https://openreview.net/forum?id=rJlf_RVKwr
[sensible_adv_x_link]:https://drive.google.com/open?id=1cl-NcOYGqQe7cETLqeTdNVcDbt_SYn8L
[BAT]:https://arxiv.org/abs/1811.10716
[BAT_code_link]:https://github.com/wjyouch/bilateral-adversarial-training
[bilater_adv_x_link]:https://drive.google.com/open?id=1cl-NcOYGqQe7cETLqeTdNVcDbt_SYn8L
[GCE_paper]:https://arxiv.org/pdf/1903.09799.pdf
[GCE_github]:https://github.com/henry8527/GCE