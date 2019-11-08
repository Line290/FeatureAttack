##Feature Attack

### *Important
 90% codes copy from [FeatureScatter][FS_github]
 
### My work
Created Feature Attack, and it's stronger than PGD attack or CW attack w.r.t [Feature Scatter][FS_paper] and [Adversarial Interpolation Training][Adv_inter_paper].

### Reference Model
Model trained on CIFAR10: [FS][FS_model] and [Adv_inter][Adv_inter_model]
### Evaluate
```bash
sh fs_eval_feature_attack.sh
```

### Result

| Attack type     | clean | FGSM | PGD20 | CW20 | FeatureAttack100       |
|-----------------|-------|------|-------|------|------------------------|
| Feature Scatter | 90.3  | 78.4 | 71.1  | 62.4 | 37.2(1500 test images) |
| Adv_inter       | 90.5  | 78.1 | 74.4  | 69.5 | 37.9(2000 test images) |








[FS_github]https://github.com/Haichao-Zhang/FeatureScatter
[FS_paper]https://arxiv.org/pdf/1907.10764.pdf
[Adv_inter_paper]https://openreview.net/pdf?id=Syejj0NYvr
[FS_model]https://drive.google.com/open?id=1TCw1uVrAikOZIObHfALE-FuXXa7UKDDo
[Adv_inter_model]https://drive.google.com/open?id=1ak-Qovkra3oIqukAWc32rLJcAPkdpN79