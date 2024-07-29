conf_ppo = {'max_episodes': 300000,
            'update_timestep': 60*10,   # max time env_evs 180 sec, 60 -> one episode of actions
            'action_std' : 0.1,
            'K_epochs': 80,               # update policy for K epochs
            'eps_clip': 0.2,              # clip parameter for PPO
            'gamma' : 1.0,                # discount factor
            'lr': 1e-5,                 # parameters for Adam optimizer
            'betas' : (0.9, 0.999),   
            'random_seed': None,
            'lam_a' : 0,
            'normalize_rewards': True, # reward 정규화
            'nn_type' : 'tanh'}

# update_timestep : policy 업데이트 주기 60 : 한 에피소드에서의 행동 수, 10 : 10번 에피소드를 반복