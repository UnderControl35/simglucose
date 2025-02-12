#!/home/guleserhocam/miniconda3/envs/simbg/bin/python

from datetime import date
import gym
from numpy import array
from gym.envs.registration import register
import os, sys, keyboard
from termcolor import colored
from pygame.locals import *
import pickle as pkl
import argparse
from simglucose.controller.base import Action
from simglucose.controller.pid_ctrller_v2 import PIDController
import pandas as pd
from datetime import datetime
import numpy as np
from simglucose.analysis.risk import risk_index

class PIDAction:
    def __init__(self, P=1, I=0, D=0, target=0):
        self.P = P
        self.I = I
        self.D = D
        self.target = target
        self.integrated_state = 0
        self.prev_state = 0

    def policy(self, current_act):
        sampling = 3

        # BG is the only state for this PID controller
        control = current_act
        error = self.target - control
        control_input = self.P * error + self.I * self.integrated_state + self.D * (self.prev_state - control)

        # update the states
        self.prev_state = control
        self.integrated_state += error

        # # return the actionq
        action = control_input
        if action <0: action=0
        if self.target<0: self.target=0
            
        print(f"Target Action: {(self.target)} \t Action: {(action)}")
    
        #print("Target:%.3f"%self.target)
        return action
        
    def reset(self):
        self.prev_state = 0
        self.integrated_state = 0
        
    def key_callback(self):
        if keyboard.is_pressed('up'):
            self.target +=0.5
            #self.target+=0.5
        if keyboard.is_pressed('down'):
            self.target -=0.5


def main():
    n_trajectory = 0

    # Register gym environment. By specifying kwargs,
    # you are able to choose which patient to simulate.
    # patient_name must be 'adolescent#001' to 'adolescent#010',
    # or 'adult#001' to 'adult#010', or 'child#001' to 'child#010'
    register(
        id='simglucose-adult-v1',
        entry_point='simglucose.envs:T1DSimEnv',
        kwargs={'patient_name': 'adult#001',}
    )

    env = gym.make('simglucose-adult-v1')
    observation = env.reset()
    action_ref = 0
    act_obj = PIDAction(P=args.pid_tune[0], 
                        I=args.pid_tune[1], 
                        D=args.pid_tune[2], target=action_ref)

    # 0: load, 1: collect
    if args.collect:
        memory = []
        for e in range(args.episodes):
            obs_record = []
            next_obs_record = []
            rew_record = []
            action_record = []
            dones = []
            timestamps = []

            pid_controller = PIDController(P=-1.74e-04, I=-1e-09, D=-1e-015, target=120)
            pid_controller.reset()

            for t in range(args.timesteps):
                env.render(mode='human')
                action = 0
                if args.collect: 
                    if args.collect_type == 'manual':
                        act_obj.key_callback()
                        action = act_obj.policy(action)
                    elif args.collect_type == 'pid':
                        if t==0: pass
                        else: 
                            act = pid_controller.policy(observation, reward, done, **info)
                            basal_act, bolus_act = act.basal, act.bolus
                            action = basal_act + bolus_act
                    else:
                        raise NameError("Please select correct controller!")

                # Action in the gym environment is a scalar
                # representing the basal insulin, which differs from
                # the regular controller action outside the gym
                # environment (a tuple (basal, bolus)).qq
                # In the perfect situation, the agent should be able
                # to control the glucose only through basal instead
                # of asking patient to take bolus

                next_observation, reward, done, info = env.step(action)
                insulin_value = env.env.insulin_hist[-1]
                _, _, risk = risk_index([observation.CGM], 1)
                _, _, next_risk = risk_index([next_observation.CGM], 1)

                obs_record.append([observation.CGM])
                next_obs_record.append([next_observation.CGM])
                action_record.append([insulin_value])
                rew_record.append(reward)
                dones.append(done)
                timestamps.append(t)

                observation = next_observation

                if done or t==(args.timesteps-1):
                    #GAIL uses fixed timestep for all records
                    if t==(args.timesteps-1):
                        memory.append({'observations': np.array(obs_record, dtype=np.float32), 
                        'next_observations':np.array(next_obs_record, dtype=np.float32), 
                        'actions':np.array(action_record, dtype=np.float32), 'rewards':np.array(rew_record, dtype=np.float32), 
                        'terminals':np.array(dones)})

                    print("Episode finished after {} timesteps".format(t + 1))
                    observation = env.reset()
                    act_obj.reset()
                    pid_controller.reset()
                    env.close()
                    action_ref = 0
                    n_trajectory+=1

        # #Save as csv file
        # df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in df.items() ]))
        # df.to_csv(args.save_path + f"_eps_{n_trajectory}" + "-%s.csv"%datetime.now().replace(second=0, microsecond=0))
        # df2 = pd.DataFrame(data=memory)
        # df2.to_csv(args.save_path + "_Memory" + f"_eps_{n_trajectory}" + "-%s.csv"%datetime.now().replace(second=0, microsecond=0))

        #Save as pickle file
        with open(args.save_path + "DATA" + f"_eps_{n_trajectory}" + "-%s.pkl"%datetime.now().replace(second=0, microsecond=0), 'wb') as handle:
            pkl.dump(memory, handle, protocol=pkl.HIGHEST_PROTOCOL)

        print('Trajectories are experimented: ', n_trajectory)
        print('Trajectories are saved:', len(memory))

    else:
        file_flag=0
        if not len(os.listdir(args.save_path))==0:
            for file in os.listdir(args.save_path):
                if file.endswith(".pkl") and file_flag==0:
                    path = input("Please input explicit import path of data:")
                    with open(os.path.join(args.save_path, path), 'rb') as handle:
                        memory = pkl.load(handle)
                    print(os.path.join(args.save_path, path) + " is loaded!")
                    print(colored(f"Length Memory Trajectories: {len(memory)}",'red','on_white'))
                    print(colored(f"Trajectory timesteps: {len(memory[0]['observations'])}",'red','on_white'))
                    file_flag = 1
                else:
                    raise FileNotFoundError("Inconsistent path for the file or loaded!")
        else:
            raise FileNotFoundError("There is no any dataset in folder!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--collect', type=bool, default= 1)
    parser.add_argument('--collect_type', type=str, default= 'manual', help="Select option manual/pid")
    parser.add_argument('--save_path', type=str, default= '/home/berk/VS_Project/simglucose/examples/trajectories/')
    parser.add_argument('--pid_tune', nargs="+", default= [1e-1, 0, 0])
    parser.add_argument('--episodes', type=int, default= 10)
    parser.add_argument('--timesteps', type=int, default= 480*5) #5 Days

    args = parser.parse_args()
    main()