#liaoyihang

## model_free
写了几种常见的model free算法

-Deep Q Learning

* DQN 广泛应用于离散动作空间的环境中（例如游戏等）， 通过deep neural network找到当前状态下预测q值
最大的动作。

-Vanilla Policy Gradient

-Deep Deterministic Policy Gradient

-Proximal Policy Optimizationptimization

-TD3

* policy gradient的算法都可以用于连续动作空间的环境中，但是训练的时候有不同的小trick会导致收敛效果
不同

## model_based
-model_ensemble

model ensemble算法仅加了一个对于环境predict的模型。

-model assisted

训练时，根据训练的reward方差，调整训练时所采用的样本，方差越大，则采用越多从model中generate的训练信息
反之则采用环境所给予的训练数据

modelbased算法相较于model free算法最主要的区别是在于agent有无对于环境的估计，对于model base的agent
由于他们自己对于环境有一个估计，从而使得在训练中可以根据模型自己产生新的训练数据（通过rolloutrollout 的方法）
所以，modelbase算法相较于modelfree算法sample efficiency更高，在有限的训练数据中会有更好的训练效果，但是
由于训练的是两个模型，会产生两次的训练误差。

## exploration & exploitation trade off

简单的例子介绍exploration和exploitation：
比如吃饭有两个选择： 1.去一家新的餐厅（exploration） 2.去已知最喜欢的餐厅（exploitationexploitation）
在强化学习训练中也是这样， 是否需要agent去探索可能存在的更优策略，或者是在继续优化已知的策略， 哪个效果更好？

在模型训练时，需要对exploration 和 exploitation进行平衡， 如果explore 太多了会导致模型的训练变得异常缓慢，
如果eploit 太多则可能导致模型 stuck at local optimum，模型则可能是局部最优而不是全局最优。

### 多臂老虎机的问题

运用了upper confidence bound来选择哪个bandits，可以从结果看到ucb会比greedy在多臂老虎机的环境中有更好的结果

### contextual bandits 问题

### 蒙特卡洛树搜索

蒙特卡洛树搜索是一颗搜索树，它经历下面几个过程：

选择（Selection）
扩展 (expansion)
模拟（Simluation）
回溯（Backpropagation）

