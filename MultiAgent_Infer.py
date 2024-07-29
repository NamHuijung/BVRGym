import argparse
import torch
import numpy as np
from tqdm import tqdm
from enum import Enum
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from jsb_gym.environmets import MultiAgent
from jsb_gym.RL.ppo import PPO, Memory
from jsb_gym.environmets.config import BVRGym_1v1
from jsb_gym.TAU.config import aim_BVRGym, f16_BVRGym

class Maneuvers(Enum):
    Evasive = 0
    Crank = 1
    
def load_model(model_path, state_dim, action_dim, conf_ppo, use_gpu=True):
    ppo = PPO(state_dim, action_dim, conf_ppo, use_gpu=use_gpu)
    ppo.policy.load_state_dict(torch.load(model_path))
    ppo.policy.eval()
    return ppo

def run_inference(args):
    if args['track'] == 'MA':
        from jsb_gym.RL.config.ppo_1v1 import conf_ppo 
        env = MultiAgent.AirCombat_1v1(BVRGym_1v1, args, aim_BVRGym, f16_BVRGym)
    
    writer = SummaryWriter('/home/nhj/BVRGym/MY/MA/BVRGym/LOG_FILE/tb_logs/test' + args['track'])
    memory = Memory()
    model_path = args['model_path']
    state_dim = env.observation_space
    action_dim = env.action_space.shape[1]
    
    ppo = load_model(model_path, state_dim, action_dim, conf_ppo, use_gpu=True)
    time_step = 0
    
    rewards = []
    maneuver = Maneuvers.Evasive
    with torch.no_grad():
        for episode in tqdm(range(1, args['Eps']+1)):
            state, state_block = env.reset(rand_state_f16=False, rand_state_sam=False)
            state = np.concatenate((state, state_block['sam1'][0], state_block['sam2'][0]))
            done = False
            total_reward = 0
            step_count = 0
            
            while not done:
                action = ppo.select_action(state, memory, gready=True)
                state, state_block, reward, done, _ = env.step(action, action_type=maneuver.value)
                state = np.concatenate((state, state_block['sam1'][0], state_block['sam2'][0]))
                total_reward += reward
                step_count += 1
                if args['vizualize']:
                    env.render()  # 시각화 옵션이 켜져 있을 때만 시각화

            rewards.append(total_reward)
            print(f'Episode {episode + 1}, Reward: {total_reward}')
    
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Inference Rewards over Episodes')
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vizualize", action='store_true', help="Render in FG")
    parser.add_argument("-track", "--track", type=str, help="Tracks: MA", default=' ')
    parser.add_argument("-Eps", "--Eps", type=int, help="Number of episodes to run", default=int(1e3))
    parser.add_argument("-eps", "--eps", type=int, help="Number of episodes to run", default=5)
    parser.add_argument("-model_path", "--model_path", type=str, help="Path to the model file", required=True)
    args = vars(parser.parse_args())

    run_inference(args)
    
# python inference.py -track MA -Eps 100 -model_path /path/to/model.pth
