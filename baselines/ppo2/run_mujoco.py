#!/usr/bin/env python3
import numpy as np
from baselines.common.cmd_util import mujoco_arg_parser, arg_parser
import os
import pybullet_envs
from baselines import bench, logger


def train(env_id, num_timesteps, seed):
    from baselines.common import set_global_seeds
    from baselines.common.vec_env.vec_normalize import VecNormalize
    from baselines.ppo2 import ppo2
    from baselines.ppo2.policies import MlpPolicy
    import gym
    import tensorflow as tf
    from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
    ncpu = 1
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    tf.Session(config=config).__enter__()

    def make_env():
        env = gym.make(env_id)
        env = bench.Monitor(env, logger.get_dir(), allow_early_resets=True)
        return env

    env = DummyVecEnv([make_env])
    env = VecNormalize(env)

    set_global_seeds(seed)
    policy = MlpPolicy
    model = ppo2.learn(policy=policy, env=env, nsteps=2048, nminibatches=32,
                       lam=0.95, gamma=0.99, noptepochs=10, log_interval=1,
                       ent_coef=0.0,
                       lr=3e-4,
                       cliprange=0.2,
                       total_timesteps=num_timesteps)

    return model, env


def main():

    parser = arg_parser()
    parser.add_argument('--env', help='environment ID', type=str, default='Reacher-v2')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--policy', help='policy ID', type=str, default='PPO')
    parser.add_argument('--num-timesteps', type=int, default=int(1e6))
    parser.add_argument('--dir', type=str, default='data')
    parser.add_argument('--lr', type=float, default=3e-4, help="Learning rate")
    parser.add_argument('--play', default=False, action='store_true')


    import time
    args = parser.parse_args()
    exp_name = '_'.join([args.policy,args.env])

    ymd_time = time.strftime("%Y-%m-%d_")
    relpath = ''.join([ymd_time, exp_name])

    hms_time = time.strftime("%Y-%m-%d_%H-%M-%S")
    subfolder = ''.join([hms_time, '-', exp_name, '_seed_', str(args.seed)])

    args.dir = os.path.join('data',relpath, subfolder)

    logger.configure(args.dir)
    model, env = train(
        args.env,
        num_timesteps=args.num_timesteps,
        seed=args.seed
    )

    if args.play:
        logger.log("Running trained model")
        obs = np.zeros((env.num_envs,) + env.observation_space.shape)
        obs[:] = env.reset()
        while True:
            actions = model.step(obs)[0]
            obs[:]  = env.step(actions)[0]
            env.render()


if __name__ == '__main__':
    main()
