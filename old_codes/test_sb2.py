import gym
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import A2C
from gym.envs.registration import register
from stable_baselines.common.env_checker import check_env
import pickle as pkl
import argparse
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def custom_reward(BG_last_hour):
    if BG_last_hour[-1] > 180:
        return -1
    elif BG_last_hour[-1] < 70:
        return -2
    else:
        return 1


def main():

    register(
            id='simglucose-adult-v1',
            entry_point='simglucose.envs:T1DSimEnv',
            kwargs={'patient_name': 'adult#001',
                    'reward_fun': custom_reward}
        )

    env = gym.make('simglucose-adult-v1')
    #check_env(env)
    obs = env.reset()

    if not args.test:
        model = A2C(MlpPolicy, env, verbose=1)
        model.learn(total_timesteps=args.timesteps)
        model.save(args.save_path + 'a2c_simbg')

    elif args.test:
        model = A2C.load(args.save_path + 'a2c_simbg')
        done=0
        t=0
        while not done:
            env.render('human')
            action, _states = model.predict(obs)
            obs, rewards, done, info = env.step(action)
            t+=1

            if done:
                print(f"\x1b[6;30;42m Episode length:{t} \x1b[0m")
                env.close()
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--collect', type=bool, default= 0)
    parser.add_argument('--save_path', type=str, default= '/home/berk/VS_Project/simglucose/examples/trajectories/')
    parser.add_argument('--pid_tune', nargs="+", default= [0.1, 0, 0.1])
    parser.add_argument('--episodes', type=int, default= 2)
    parser.add_argument('--timesteps', type=int, default= int(1e6))
    parser.add_argument('--test', type=bool, default= 0)

    args = parser.parse_args()
    main()