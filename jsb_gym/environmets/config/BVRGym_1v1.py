import numpy as np

version = 1

aim1 = {'bearing': 0,
        'distance': 110e3,
        'vel': 300,
        'alt': 12e3}

aim2 = {'bearing': 0,
        'distance': 110e3,
        'vel': 300,
        'alt': 12e3}

aim1r = {'bearing': 0,
        'distance': 110e3,
        'vel': 300,
        'alt': 12e3}

# aim2r = {'bearing': 0,
#         'distance': 110e3,
#         'vel': 300,
#         'alt': 12e3}

sam1 = {'bearing': 90,
        'distance': 90e3,
        'vel': 300,
        'alt': 11e3}

sam2 = {'bearing': 270,
        'distance': 90e3,
        'vel': 300,
        'alt': 11e3}

aim = {'aim1': aim1, 'aim2': aim2}

# aimr = {'aim1r': aim1r, 'aim2r': aim2r}
aimr = {'aim1r': aim1r}

sam = {'sam1': sam1, 'sam2': sam2}

# random location expects a list 
sam1_rand = {'bearing':  [0, 360],
             'distance': [70e3, 80e3],
             'vel':      [290, 340],
             'alt':      [10e3, 12e3]}

sam2_rand = {'bearing':  [0, 360],
             'distance': [70e3, 80e3],
             'vel':      [290, 340],
             'alt':      [10e3, 12e3]}

sam_rand = {'sam1': sam1_rand, 'sam2': sam2_rand}

general = {
        'env_name': 'MultiAgent',
        'f16_name': 'f16',
        'f16r_name': 'f16r',
        'sim_time_max': 60*16,           
        'r_step' : 30,
        'fg_r_step' : 1,
        'missile_idle': False,
        'scale': True,
        'rec':False}

states= {
        'obs_space': 10,
        'act_space': 2,
        'update_states_type': 1,
        'f16_state_dim': 15,
        'sam_state_dim': 10
}

logs= {'log_path': '/home/nhj/BVRGym/MY/MA/BVRGym/LOG_FILE/logs/Multi',
       'save_to': '/home/nhj/BVRGym/MY/MA/BVRGym/LOG_FILE/plots/Multi'}

sf = {  'd_min': 20e3,
        'd_max': 120e3,
        't': general['sim_time_max'],
        'mach_max': 2,
        'alt_min': 3e3,
        'alt_max': 12e3,
        'head_min': 0,
        'head_max': 360,
        'd_max_reward': 20e3,
        'aim_vel0_min': 280, 
        'aim_vel0_max': 320, 
        'aim_alt0_min': 9e3, 
        'aim_alt0_max': 11e3
}


f16 = { 'lat':      58.3,
        'long':     18.0,
        'vel':      350,
        'alt':      10e3,
        'heading' : 0}

f16r = { 'lat':      59.0,
         'long':     18.0,
         'vel':      350,
         'alt':      10e3,
         'heading' : 180}