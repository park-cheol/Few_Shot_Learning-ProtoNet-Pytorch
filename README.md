# Prototypical Networks for Few Shot Learning in PyTorch

paper: https://arxiv.org/abs/1703.05175

code:
1. https://github.com/orobix/Prototypical-Networks-for-Few-shot-Learning-PyTorch/blob/master/src/prototypical_loss.py

2. https://github.com/jakesnell/prototypical-networks/blob/master/protonets/models/few_shot.py

# ProtoNet
![1](https://user-images.githubusercontent.com/76771847/147207414-34dd6318-d105-4fb3-8b9b-05f7a1d37837.png)

- ***Episode Training***: Test와 비슷한 상황으로 학습시키기 위해 Train Dataset에서 N개의 Class와 K개의 Examples를
RandomSampling(N way K shot) 한 후 남은 Examples들 중에서 한 개를 RandomSampling 하여 Query Examples를 만든 후 학습

- ***ProtoNet***: Space에 Mapping 학습 -> embedding된 K개 Support Examples(Shot)들을 평균(c1, c2, c3) ->
embedding된 Query point와 c1, c2, c3을 Euclidean Distance를 구한 후 유사성을 판단

# Implement

1. **5 shot 5 way[Acc: 99.7]**
> python main.py --gpu 0

2. **1 shot 20 way[Acc: 94.8]**
> python main.py --gpu 0 --num-support-train 1 --num-support-valid 1 --classes-per-it-valid 20
>


