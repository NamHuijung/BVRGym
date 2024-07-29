import os
import json
import argparse, time
from jsb_gym.environmets import MultiAgent
import numpy as np
from enum import Enum
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch
import multiprocessing
import torch.multiprocessing as mp
from jsb_gym.RL.ppo import Memory, PPO
from jsb_gym.environmets.config import BVRGym_1v1
from numpy.random import seed
from jsb_gym.TAU.config import aim_BVRGym, f16_BVRGym

def init_pool():
    seed()

class Maneuvers(Enum):
    Evasive = 0
    Crack = 1

def runPPO(args):
    if args['track'] == 'MA':
        from jsb_gym.RL.config.ppo_1v1 import conf_ppo 
        env = MultiAgent.AirCombat_1v1(BVRGym_1v1, args, aim_BVRGym, f16_BVRGym)
        torch_save = '/home/nhj/BVRGym/MY/MA/BVRGym/LOG_FILE/'
        state_scale = 1
        
    writer = SummaryWriter('/home/nhj/BVRGym/MY/MA/BVRGym/LOG_FILE/tb_logs/' + args['track'])
    state_dim = env.observation_space
    action_dim = env.action_space.shape[1]
    memory = Memory()
    ppo = PPO(state_dim*state_scale, action_dim, conf_ppo, use_gpu = True)    
    pool = multiprocessing.Pool(processes=int(args['cpu_cores']), initializer=init_pool)
    
    best_reward = -np.inf
    
    for i_episode in tqdm(range(1, args['Eps']+1)):
        ppo_policy = ppo.policy.state_dict()    
        ppo_policy_old = ppo.policy_old.state_dict()
        input_data = [(args, ppo_policy, ppo_policy_old, conf_ppo, state_scale, k)for k in range(args['cpu_cores'])]
        running_rewards = []
        tb_obs = []
        eps_log_data = []
        
        results = pool.map(train, input_data)
        for idx, tmp in enumerate(results):
            #print(tmp[5])
            #writer.add_scalar("running_rewards" + str(idx), tmp[5], i_episode)
            memory.actions.extend(tmp[0])
            memory.states.extend(tmp[1])
            memory.logprobs.extend(tmp[2])
            memory.rewards.extend(tmp[3])
            memory.is_terminals.extend(tmp[4])
            running_rewards.append(tmp[5])
            tb_obs.append(tmp[6])
            eps_log_data.extend(tmp[7])
            
        ppo.set_device(use_gpu=True)
        ppo.update(memory, to_tensor=True, use_gpu= True)
        memory.clear_memory()
        ppo.set_device(use_gpu=True)
        torch.cuda.empty_cache()
        
        total_rewards = sum(running_rewards)/len(running_rewards)
        writer.add_scalar("running_rewards", total_rewards, i_episode)

        # 각 키의 값을 키의 개수로 나누어 평균을 계산
        tb_obs0 = None
        for i in tb_obs:
            if tb_obs0 is None:
                tb_obs0 = {key: 0 for key in i}  # tb_obs0 초기화
                count = {key: 0 for key in i}    # 각 키의 개수를 세기 위한 딕셔너리 초기화
            for key in i:
                if key in tb_obs0:
                    tb_obs0[key] += i[key]
                    count[key] += 1  # 키의 개수 증가
                else:
                    tb_obs0[key] = i[key]
                    count[key] = 1
        for key in tb_obs0:
            tb_obs0[key] = tb_obs0[key]/count[key]
            writer.add_scalar(key, tb_obs0[key], i_episode)
            
        if i_episode % 100 == 0: # save 
            torch.save(ppo.policy.state_dict(), torch_save + 'Multi_' + str(i_episode) + '.pth')
        
        if total_rewards > best_reward:
            best_reward = total_rewards
            torch.save(ppo.policy.state_dict(), torch_save + 'best_model.pth')
            
        # 에피소드 별 로그 데이터 파일 저장
        log_file = os.path.join(torch_save+'logs/ep500_2/', f'episode_{i_episode}_log.json')
        with open(log_file, 'w') as f:
            json.dump(eps_log_data, f, indent=4)

    pool.close()
    pool.join()
    torch.save(ppo.policy.state_dict(), torch_save + 'MA.pth')
    
def train(args):
    # print(f"len(args[1]) : {args[1]}")
    cpu_id = args[5]
    if args[0]['track'] == 'MA':
        env = MultiAgent.AirCombat_1v1(BVRGym_1v1, args[0], aim_BVRGym, f16_BVRGym)
        
    maneuver = Maneuvers.Evasive
    memory = Memory()
    state_dim = env.observation_space
    action_dim = env.action_space.shape[1]
    ppo = PPO(state_dim* args[4], action_dim, args[3], use_gpu=True)

    ppo.policy.load_state_dict(args[1])
    ppo.policy_old.load_state_dict(args[2])

    ppo.policy.eval()
    ppo.policy_old.eval()
    running_reward = 0.0
    
    log_data = []
    
    for i_episode in range(1, args[0]['eps']+1):
        action = np.zeros(3)
        # using string comparison, not the best, that is why I am keeping it short for now
        if args[0]['track'] == 'MA':
            state, state_block = env.reset(rand_state_f16=False, rand_state_sam=True)
            state = np.concatenate((state, state_block['sam1'][0], state_block['sam2'][0]))
            # max thrust 
            action[2] = 0.0
        
        done = False
        while not done:
            # heading [-1, 1] altitude [-1, 1] thrust [-1, 1]
            act = ppo.select_action(state, memory)
            action[0] = act[0]
            action[1] = act[1]
            
            if args[0]['track'] == 'MA':
                state, state_block, reward, done, _ = env.step(action, action_type=maneuver.value, blue_armed=True, red_armed=True)
                state = np.concatenate((state, state_block['sam1'][0], state_block['sam2'][0]))
                
            memory.rewards.append(reward)
            memory.is_terminals.append(done)
            
            # F16, AIM, SAM, ENV log
            log_entry = {
                "cpu_id": cpu_id,
                "sim_time": env.f16.get_sim_time_sec(),
                "blue_alive": env.f16_alive,
                "red_alive": env.f16r_alive,
                "Blue_ground": env.reward_f16_hit_ground,
                "Red_ground": env.reward_f16r_hit_ground,
                "reward": reward,
                "f16_phi": env.f16.get_phi(),
                "f16_theta": env.f16.get_theta(),
                "f16_psi": env.f16.get_psi(),
                "f16_lat": env.f16.get_lat_gc_deg(),
                "f16_long": env.f16.get_long_gc_deg(),
                "f16_alt": env.f16.get_altitude(),
                "f16_mach": env.f16.get_Mach(),
                "sam1_alt": env.sam_block['sam1'].get_altitude(),
                "sam1_mach": env.sam_block['sam1'].get_Mach(),
                "sam1_lat": env.sam_block['sam1'].get_lat_gc_deg(),
                "sam1_long": env.sam_block['sam1'].get_long_gc_deg(),
                "sam1_psi": env.sam_block['sam1'].get_psi(),
                "sam1_target_pos": env.sam_block['sam1'].position_tgt_NED_norm,
                "sam2_alt": env.sam_block['sam2'].get_altitude(),
                "sam2_mach": env.sam_block['sam2'].get_Mach(),
                "sam2_lat": env.sam_block['sam2'].get_lat_gc_deg(),
                "sam2_long": env.sam_block['sam2'].get_long_gc_deg(),
                "sam2_psi": env.sam_block['sam2'].get_psi(),
                "sam2_target_pos": env.sam_block['sam2'].position_tgt_NED_norm,
                "f16r_phi": env.f16r.get_phi(),
                "f16r_theta": env.f16r.get_theta(),
                "f16r_psi": env.f16r.get_psi(),
                "f16r_lat": env.f16r.get_lat_gc_deg(),
                "f16r_long": env.f16r.get_long_gc_deg(),
                "f16r_alt": env.f16r.get_altitude(),
                "f16r_mach": env.f16r.get_Mach(),
                "aim1_active": env.aim_block['aim1'].active,
                "aim1_alive": env.aim_block['aim1'].alive,
                "aim1_target_lost": env.aim_block['aim1'].target_lost,
                "aim1_target_hit": env.aim_block['aim1'].target_hit,
                "aim2_active": env.aim_block['aim2'].active,
                "aim2_alive": env.aim_block['aim2'].alive,
                "aim2_target_lost": env.aim_block['aim2'].target_lost,
                "aim2_target_hit": env.aim_block['aim2'].target_hit,
                "aim1r_active": env.aimr_block['aim1r'].active,
                "aim1r_alive": env.aimr_block['aim1r'].alive,
                "aim1r_target_lost": env.aimr_block['aim1r'].target_lost,
                "aim1r_target_hit": env.aimr_block['aim1r'].target_hit
            }
            log_data.append(log_entry)
            

        running_reward += reward 
        
    running_reward = running_reward/args[0]['eps']
    # tensorboard 
    if args[0]['track'] == 'MA':
        tb_obs = get_tb_obs_dog(env)
    
    actions = [i.cpu().detach().numpy() for i in memory.actions]  # 수정된 부분
    states = [i.cpu().detach().numpy() for i in memory.states]    # 수정된 부분
    logprobs = [i.cpu().detach().numpy() for i in memory.logprobs] # 수정된 부분
    rewards = [i for i in memory.rewards]
    #print(rewards)
    is_terminals = [i for i in memory.is_terminals]     
    return [actions, states, logprobs, rewards, is_terminals, running_reward, tb_obs, log_data]

def get_tb_obs_dog(env):
    tb_obs = {}
    tb_obs['Blue_ground'] = env.reward_f16_hit_ground
    tb_obs['Red_ground'] = env.reward_f16r_hit_ground
    tb_obs['maxTime'] = env.reward_max_time

    tb_obs['Blue_alive'] = env.f16_alive
    tb_obs['Red_alive'] = env.f16r_alive

    tb_obs['aim1_active'] = env.aim_block['aim1'].active
    tb_obs['aim1_alive'] = env.aim_block['aim1'].alive
    tb_obs['aim1_target_lost'] = env.aim_block['aim1'].target_lost
    tb_obs['aim1_target_hit'] = env.aim_block['aim1'].target_hit

    tb_obs['aim2_active'] = env.aim_block['aim2'].active
    tb_obs['aim2_alive'] = env.aim_block['aim2'].alive
    tb_obs['aim2_target_lost'] = env.aim_block['aim2'].target_lost
    tb_obs['aim2_target_hit'] = env.aim_block['aim2'].target_hit

    tb_obs['aim1r_active'] = env.aimr_block['aim1r'].active
    tb_obs['aim1r_alive'] = env.aimr_block['aim1r'].alive
    tb_obs['aim1r_target_lost'] = env.aimr_block['aim1r'].target_lost
    tb_obs['aim1r_target_hit'] = env.aimr_block['aim1r'].target_hit

    # tb_obs['aim2r_active'] = env.aimr_block['aim2r'].active
    # tb_obs['aim2r_alive'] = env.aimr_block['aim2r'].alive
    # tb_obs['aim2r_target_lost'] = env.aimr_block['aim2r'].target_lost
    # tb_obs['aim2r_target_hit'] = env.aimr_block['aim2r'].target_hit

    # Additional information
    tb_obs['f16_lat'] = env.f16.get_lat_gc_deg()
    tb_obs['f16_long'] = env.f16.get_long_gc_deg()
    tb_obs['f16_alt'] = env.f16.get_altitude()
    tb_obs['f16_sim_time'] = env.f16.get_sim_time_sec()
    tb_obs['f16_mach'] = env.f16.get_Mach()
    tb_obs["f16_phi"] = env.f16.get_phi()
    tb_obs["f16_theta"] = env.f16.get_theta()
    tb_obs["f16_psi"] = env.f16.get_psi()

    tb_obs['f16r_lat'] = env.f16r.get_lat_gc_deg()
    tb_obs['f16r_long'] = env.f16r.get_long_gc_deg()
    tb_obs['f16r_alt'] = env.f16r.get_altitude()
    tb_obs['f16r_sim_time'] = env.f16r.get_sim_time_sec()
    tb_obs['f16r_mach'] = env.f16r.get_Mach()
    tb_obs["f16r_phi"] = env.f16r.get_phi()
    tb_obs["f16r_theta"] = env.f16r.get_theta()
    tb_obs["f16r_psi"] = env.f16r.get_psi()
    
    # SAM-sam1
    tb_obs['sam1_lat'] = env.sam_block['sam1'].get_lat_gc_deg()
    tb_obs['sam1_long'] = env.sam_block['sam1'].get_long_gc_deg()
    tb_obs['sam1_alt'] = env.sam_block['sam1'].get_altitude()
    tb_obs['sam1_mach'] = env.sam_block['sam1'].get_Mach()
    tb_obs['sam1_target_pos'] = env.sam_block['sam1'].position_tgt_NED_norm
    tb_obs['sam1_sim_time'] = env.sam_block['sam1'].get_sim_time_sec()
    tb_obs['sam1_target_hit'] = env.sam_block['sam1'].target_hit
    tb_obs["sam1_psi"] = env.sam_block['sam1'].get_psi()
    # SAM-sam2
    tb_obs['sam2_lat'] = env.sam_block['sam2'].get_lat_gc_deg()
    tb_obs['sam2_long'] = env.sam_block['sam2'].get_long_gc_deg()
    tb_obs['sam2_alt'] = env.sam_block['sam2'].get_altitude()
    tb_obs['sam2_mach'] = env.sam_block['sam2'].get_Mach()
    tb_obs['sam2_target_pos'] = env.sam_block['sam2'].position_tgt_NED_norm
    tb_obs['sam2_sim_time'] = env.sam_block['sam2'].get_sim_time_sec()
    tb_obs['sam2_target_hit'] = env.sam_block['sam2'].target_hit
    tb_obs["sam2_psi"] = env.sam_block['sam2'].get_psi()

    if env.aimr_block['aim1r'].active:
        tb_obs['aim1r_lat'] = env.aimr_block['aim1r'].get_lat_gc_deg()
        tb_obs['aim1r_long'] = env.aimr_block['aim1r'].get_long_gc_deg()
        tb_obs['aim1r_alt'] = env.aimr_block['aim1r'].get_altitude()
        tb_obs['aim1r_sim_time'] = env.aimr_block['aim1r'].get_sim_time_sec()

    # if env.aimr_block['aim2r'].active:
    #     tb_obs['aim2r_lat'] = env.aimr_block['aim2r'].get_lat_gc_deg()
    #     tb_obs['aim2r_long'] = env.aimr_block['aim2r'].get_long_gc_deg()
    #     tb_obs['aim2r_alt'] = env.aimr_block['aim2r'].get_altitude()
    #     tb_obs['aim2r_sim_time'] = env.aimr_block['aim2r'].get_sim_time_sec()

    if env.aim_block['aim1'].active:
        tb_obs['aim1_lat'] = env.aim_block['aim1'].get_lat_gc_deg()
        tb_obs['aim1_long'] = env.aim_block['aim1'].get_long_gc_deg()
        tb_obs['aim1_alt'] = env.aim_block['aim1'].get_altitude()
        tb_obs['aim1_sim_time'] = env.aim_block['aim1'].get_sim_time_sec()

    if env.aim_block['aim2'].active:
        tb_obs['aim2_lat'] = env.aim_block['aim2'].get_lat_gc_deg()
        tb_obs['aim2_long'] = env.aim_block['aim2'].get_long_gc_deg()
        tb_obs['aim2_alt'] = env.aim_block['aim2'].get_altitude()
        tb_obs['aim2_sim_time'] = env.aim_block['aim2'].get_sim_time_sec()

    if env.aim_block['aim1'].target_lost:
        tb_obs['aim1_MD'] = env.aim_block['aim1'].position_tgt_NED_norm

    if env.aim_block['aim2'].target_lost:
        tb_obs['aim2_lost'] = 1
        tb_obs['aim2_MD'] = env.aim_block['aim2'].position_tgt_NED_norm

    if env.aimr_block['aim1r'].target_lost:
        tb_obs['aim1r_lost'] = 1
        tb_obs['aim1r_MD'] = env.aimr_block['aim1r'].position_tgt_NED_norm

    # if env.aimr_block['aim2r'].target_lost:
    #     tb_obs['aim2r_lost'] = 1
    #     tb_obs['aim2r_MD'] = env.aimr_block['aim2r'].position_tgt_NED_norm

    return tb_obs

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)  # 추가된 부분

    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vizualize", action='store_true', help="Render in FG")
    parser.add_argument("-track", "--track", type = str, help="Tracks: M1, M2, Dog, DogR", default=' ')
    parser.add_argument("-cpus", "--cpu_cores", type = int, help="Nuber of cores to use", default= None)
    parser.add_argument("-Eps", "--Eps", type = int, help="Nuber of cores to use", default= int(1e3))
    parser.add_argument("-eps", "--eps", type = int, help="Nuber of cores to use", default= 1)
    parser.add_argument("-seed", "--seed", type = int, help="radnom seed", default= int(42))
    args = vars(parser.parse_args())

    if args['seed'] != None:
       torch.manual_seed(args['seed'])
       np.random.seed(args['seed'])
       if torch.cuda.is_available():
           torch.cuda.manual_seed(args['seed'])
           torch.cuda.manual_seed_all(args['seed'])
    runPPO(args)

# training: 
# python mainBVRGym_MultiCore.py -track M1  -cpus 10 -Eps 100000 -eps 1
# python mainBVRGym_MultiCore.py -track M2  -cpus 10 -Eps 100000 -eps 1
# python mainBVRGym_MultiCore.py -track Dog -cpus 10 -Eps 10000 -eps 1