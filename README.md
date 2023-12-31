# Hierarchical-ACT
<img alt="Badge_Pytorch" src ="https://img.shields.io/badge/PyTorch-2.0.0-%23EE4C2C.svg?style=flat-square"/> <img alt="Badge_Python" src ="https://img.shields.io/badge/Python-3.8-3776AB.svg?&style=flat-square"/>

**J. Hyeon Park**, **Wonhyuk Choi**, Sunpyo Hong, Hoseong Seo, Joonmo Ahn, Changsu Ha, Heungwoo Han, Junghyun Kwon, and Sungchul Kang "**Hierarchical Action Chunk Transformer: Learning Temporal Multimodality from Demonstrations with Fast Imitation Behavior**", IEEE International Conference on Robotics and Automation (ICRA), 2024 

## Experiment video
### Stack cups
<img src="./resource/stack_cups.gif" width="640" height="360" />
</br>

### Close pocket
<img src="./resource/close_pocket.gif" width="640" height="360" />
</br>

### Open lid
<img src="./resource/open_lid.gif" width="640" height="360" />
</br>


## List of maintainers
- Jaehyeon Park (jh_raph.park@samsung.com)
- Wonhyuk Choi (wh9.choi@samsung.com)

## Installation
```
conda create -n robot_action_learner python=3.8
conda activate robot_action_learner
pip install -r requirements.txt
```

## Train with sample data
```
python ./train.py \
    --exp_dir ./experiment \
    --data_root ./data_samples \
    --data_config ./configs/dataset/srrc_dual_frankas/stack_cups.gin \
    --model_config ./configs/model/hact_vq/srrc_dual_frankas/base.gin \
    --task_config ./configs/task/hact_vq/srrc_dual_frankas/base_local.gin \
    --gpu 0
```

- ```model.pt``` will be created in ```exp_dir```.

## Create agent
```
python ./serve.py --exp_dir ./experiment
```
- ```agent.pt``` will be created in ```exp_dir```.

## Deploy agent
```
agent = torch.jit.load("./experiment/agent.pt")
agent = agent.eval()

obs = env.reset()
while not done:
    action = agent.forward(obs, timestep=t)
    obs, reward, done, truncated, info = env.step(action)
```

## Todo
- We will open the simulation (Viper-X) and datasets soon.


## Code information
This code is implemented on the top of [ACT](https://github.com/tonyzhaozh/act).
Several codes are adopted or modified from ACT.


## Acknowledgement
All members of our remarkable robotics team: Joonmo Ahn, Rakjoon Chung, Changsu Ha, Heungwoo Han, Sunpyo Hong, Jaesik Jang, Rijeong Kang, Hosang Lee, Dongwoo Park, Hoseong Seo, Jaemin Yoon (in alphabetic order)

