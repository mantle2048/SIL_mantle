#!/usr/bin/env bash


logdir="data"
algo=$1
# envs=("BipedalWalker-v3")
# envs=("HalfCheetah-v3")
# envs=("Hopper-v3")
seeds=(0)
# 现在是　Delayed Env
envs=("HalfCheetah-v3")
# "Ant-v3" "Humanoid-v3" "Walker2d-v3")
# envs=("Walker2d-v3")
# envs=("Humanoid-v3")
# envs=("Ant-v3")
# envs=("Swimmer-v3")
# envs=("Humanoid-v3")
# envs=("AntBulletEnv-v0" "Walker2DBulletEnv-v0")

# envs=("HalfCheetahBulletEnv-v0")
# envs=("AntBulletEnv-v0")
# envs=("HopperBulletEnv-v0")
# envs=("Walker2DBulletEnv-v0")
# seeds=(5 6 9)
# envs=("HumanoidBulletEnv-v0")
# envs=("HalfCheetah-v3")
# envs=("Hopper-v3" "Walker2d-v3")

# scipy.ndimage.filters.uniform_filter1d用来滤波
# 16 15 14 13 2 20 17 up_4
# 10 12 7 9 8 6 13 14 15 11 3 20 17 25 up_2


# seed 1 173 174 175 176
for env in ${envs[@]}; do
    for seed in ${seeds[@]}; do
        python -m baselines.ppo2.run_mujoco \
            --env $env \
            --seed $seed
    done
done

# for env in ${envs[@]}; do
#     for ((i=0;i<5;i+=1)); do
#         python $1 \
#             --env $env \
#             --seed $i \
#             --device 'cuda:0' \
#             --hidden_sizes 400 300 \
#             --epochs 210 \
#             --datestamp
#     done
# done
