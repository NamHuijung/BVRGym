from jsb_gym.TAU.aircraft import F16
from jsb_gym.TAU.missiles import AIM
from jsb_gym.utils.tb_logs import Env_logs
from jsb_gym.utils.utils import toolkit, Geo
import numpy as np
from geopy.distance import geodesic
import py_trees as pt 
from py_trees.display import ascii_tree
from jsb_gym.BT.reactive_seq import ReactiveSeq

# log files if needed 
from jsb_gym.utils.tb_logs import F16_logs
from jsb_gym.utils.tb_logs import AIM_logs


class BT_utils(object):
    def __init__(self, env):
        self.env = env

    def is_missile_active(self, red_missile = False):
        active, name = self.env.is_missile_active(red_missile)
        return active, name

    def get_angle_to_enemy(self, from_red_perspective = False, offset = 0, heading_cockpit = False):
        #offset = -45
        if from_red_perspective:
            # enemy 
            fdm_tgt_lat = self.env.f16.get_lat_gc_deg()
            fdm_tgt_long = self.env.f16.get_long_gc_deg()
            # own 
            fdm_lat = self.env.f16r.get_lat_gc_deg()
            fdm_long = self.env.f16r.get_long_gc_deg()
            own_psi = self.env.f16r.get_psi()
            
        else:
            # enemy 
            fdm_tgt_lat = self.env.f16r.get_lat_gc_deg()
            fdm_tgt_long = self.env.f16r.get_long_gc_deg()
            # own 
            fdm_lat = self.env.f16.get_lat_gc_deg()
            fdm_long = self.env.f16.get_long_gc_deg()
            own_psi = self.env.f16.get_psi()

               
        ref_yaw = self.env.gtk.get_bearing(fdm_lat, fdm_long, fdm_tgt_lat, fdm_tgt_long)
        #print(ref_yaw)
        if heading_cockpit:
            ref_yaw = self.env.tk.get_heading_difference(psi_ref= ref_yaw, psi_deg= own_psi)
            return ref_yaw
        else:
            return ref_yaw + offset

    def get_distance_to_enemy(self, from_red_perspective = False):
        if from_red_perspective:
            dist = self.env.get_distance_to_enemy(fdm1=self.env.f16r, fdm2=self.env.f16, scale=False)
        else:
            dist = self.env.get_distance_to_enemy(fdm1=self.env.f16, fdm2=self.env.f16r, scale=False)
        return dist

class BVRDog_BT(object):
    def __init__(self, env, red_team = True):
        self.BTState = None
        self.BTState_old = None
        self.RootSuccess = False 
        self.root = ReactiveSeq("ReactiveSeq")
        self.heading = None
        self.altitude = None
        self.launch_missile = False
        self.use_memory = False
        self.bt_utils = BT_utils(env)
        self.red_team = red_team

        '''Missile awerness system MAW'''        
        self.MAW_own = pt.composites.Selector(name = "13", memory = self.use_memory) #3
        self.MAW_own_con = MAW_own_condition('13C', self.bt_utils, self.red_team)
        self.MAW_guide_evade_act = MAW_guide_evade_action('13A', self.bt_utils, self.red_team)
        self.MAW_own.add_children([self.MAW_own_con, self.MAW_guide_evade_act])
        
        self.MAW2 = pt.composites.Sequence(name = "12", memory = self.use_memory) #2
        self.MAW_evade_act = MAW_evade_action('12A', self.bt_utils, self.red_team)
        self.MAW2.add_children([self.MAW_own, self.MAW_evade_act])

        self.MAW = pt.composites.Selector(name = "11", memory = self.use_memory) # 1
        self.MAW_con = MAW_condition('11C', self.bt_utils, self.red_team)
        self.MAW.add_children([self.MAW_con, self.MAW2])

        '''Missile guidance'''
        self.guide = pt.composites.Selector(name = "21", memory = self.use_memory) # 1
        self.guide_own_con = MAW_own_condition('21C', self.bt_utils, self.red_team)
        self.guide_own_act = Guide_own_action('21A', self.bt_utils, self.red_team)
        self.guide.add_children([self.guide_own_con, self.guide_own_act])

        '''launch'''
        self.launch = pt.composites.Selector(name = "31", memory = self.use_memory) # 1
        self.launch_con = Launch_condition('31C', self.bt_utils, self.red_team)
        self.launch_act = Launch_action('31A', self.bt_utils, self.red_team)
        self.launch.add_children([self.launch_con, self.launch_act])

        '''pursue'''
        self.pursue = pt.composites.Selector(name= "41", memory = self.use_memory) # 1
        self.pursue_con = Pursue_condition('41C', self.bt_utils, self.red_team)
        self.pursue_act = Pursue_action('41A', self.bt_utils, self.red_team)
        self.pursue.add_children([self.pursue_con, self.pursue_act])

        '''root'''
        self.root.add_children([self.MAW, self.guide, self.launch, self.pursue])
        #tree = pt.trees.BehaviourTree(self.root)
        print(ascii_tree(self.root))

    def tick(self):
        #print('-'*10)
        self.root.tick_once()
        self.BTState = self.root.tip().name
        #print('-')
        #if self.BTState != self.BTState_old:
        if self.BTState == '13A':
            self.heading = self.MAW_guide_evade_act.heading
            self.altitude = self.MAW_guide_evade_act.altitude
            self.launch_missile = self.MAW_guide_evade_act.launch_missile
        elif self.BTState == '12A':
            self.heading = self.MAW_evade_act.heading
            self.altitude = self.MAW_evade_act.altitude
            self.launch_missile = self.MAW_evade_act.launch_missile
        elif self.BTState == '21A':
            self.heading = self.guide_own_act.heading
            self.altitude = self.guide_own_act.altitude
            self.launch_missile = self.guide_own_act.launch_missile
        elif self.BTState == '31A':
            self.heading = self.launch_act.heading
            self.altitude = self.launch_act.altitude
            self.launch_missile = self.launch_act.launch_missile
        elif self.BTState == '41A':
            self.heading = self.pursue_act.heading
            self.altitude = self.pursue_act.altitude
            self.launch_missile = self.pursue_act.launch_missile
        else:
            print('Unexpected state')
            exit()

        if self.BTState != self.BTState_old:
            print(self.BTState)
            #print(self.heading)
            #print(self.altitude)
            #print('Red:  BT launch missile ', self.launch_missile)

        self.BTState_old = self.BTState

class MAW_condition(pt.behaviour.Behaviour):
    def __init__(self, name, bt_utils, red_team):
        super(MAW_condition, self).__init__(name)
        self.bt_utils = bt_utils
        self.red_team = red_team

    def no_incomming_missile(self):
        # return status about the blue teams missiles 
        active, name = self.bt_utils.is_missile_active(red_missile = not self.red_team)
        #print(active)
        if active:
            return False
        else:
            return True

    def update(self):
        #print('Tick 11C')
        if self.no_incomming_missile():
            return pt.common.Status.SUCCESS
        else:
            return pt.common.Status.FAILURE
        
class MAW_own_condition(pt.behaviour.Behaviour):
    def __init__(self, name, bt_utils, red_team):
        super(MAW_own_condition, self).__init__(name)
        self.bt_utils = bt_utils
        self.red_team = red_team

    def no_own_active_missile(self):
        # if red team is true, check if red missile is active
        active, name = self.bt_utils.is_missile_active(red_missile = self.red_team)
        if active:
            return False
        else:
            return True

    def update(self):
        #print('Tick 21C')
        #self.feedback_message = "MAW_own_condition"
        if self.no_own_active_missile():
            return pt.common.Status.SUCCESS
        else:
            return pt.common.Status.FAILURE

class MAW_guide_evade_action(pt.behaviour.Behaviour):
    def __init__(self, name, bt_utils, red_team):
        super(MAW_guide_evade_action, self).__init__(name)
        self.bt_utils = bt_utils
        self.heading = None
        self.altitude = None
        self.launch_missile = False
        self.red_team = red_team


    def update(self):
        # evade in flank 
        self.heading = self.bt_utils.get_angle_to_enemy(from_red_perspective = self.red_team, offset = 45)
        # set 6 km altitude 
        self.altitude = 7e3
        
        #self.heading = None
        #self.altitude = None
        self.launch_missile = False
        return pt.common.Status.RUNNING

class MAW_evade_action(pt.behaviour.Behaviour):
    def __init__(self, name, bt_utils, red_team):
        super(MAW_evade_action, self).__init__(name)
        self.bt_utils = bt_utils
        self.heading = None
        self.altitude = None
        self.launch_missile = False
        self.red_team = red_team
        #offset = -180

    def update(self):
        # evade in 180 oposite direction from 
        self.heading = self.bt_utils.get_angle_to_enemy(from_red_perspective = self.red_team, offset = 180)
        # set 6 km altitude 
        self.altitude = 8e3
        self.launch_missile = False

        return pt.common.Status.RUNNING

class Guide_own_action(pt.behaviour.Behaviour):
    def __init__(self, name, bt_utils, red_team):
        super(Guide_own_action, self).__init__(name)
        self.bt_utils = bt_utils
        self.red_team = red_team
        self.heading = None
        self.altitude = None
        self.launch_missile = False
    
    def update(self):
        
        self.heading = self.bt_utils.get_angle_to_enemy(from_red_perspective = self.red_team, offset = 45)
        self.altitude = 8e3
        self.launch_missile = False
        return pt.common.Status.RUNNING

class Pursue_condition(pt.behaviour.Behaviour):
    def __init__(self, name, bt_utils, red_team):
        super(Pursue_condition, self).__init__(name)
        self.bt_utils = bt_utils
        self.red_team = red_team   

    def enemy_not_alive(self):
        return False

    def update(self):
        #self.feedback_message = "MAW_condition"
        if self.enemy_not_alive():
            return pt.common.Status.SUCCESS
        else:
            return pt.common.Status.FAILURE

class Pursue_action(pt.behaviour.Behaviour):
    def __init__(self, name, bt_utils, red_team):
        super(Pursue_action, self).__init__(name)
        self.bt_utils = bt_utils
        self.heading = None
        self.altitude = None
        self.launch_missile = False
        self.red_team = red_team

    def update(self):
        
        self.heading = self.bt_utils.get_angle_to_enemy(from_red_perspective = self.red_team, offset = 0)
        self.altitude = 10e3
        self.launch_missile = False
        return pt.common.Status.RUNNING

class Launch_condition(pt.behaviour.Behaviour):
    def __init__(self, name, bt_utils, red_team):
        super(Launch_condition, self).__init__(name)
        self.bt_utils = bt_utils
        self.red_team = red_team

    def is_in_launch_range(self):
        dist = self.bt_utils.get_distance_to_enemy(from_red_perspective = self.red_team)
        head_cockpit = self.bt_utils.get_angle_to_enemy(from_red_perspective = self.red_team, offset = 0, heading_cockpit = True)
        #print('distance: ', round(dist))
        if dist < 60e3 and abs(head_cockpit) < 35:
            return True
        else:
            return False

    def not_in_launch_position(self):
        # return status about the blue teams missile 
        if self.is_in_launch_range():
            return False
        else:
            return True

    def update(self):
        self.feedback_message = "MAW_condition"
        if self.not_in_launch_position():
            return pt.common.Status.SUCCESS
        else:
            return pt.common.Status.FAILURE

class Launch_action(pt.behaviour.Behaviour):
    def __init__(self, name, bt_utils, red_team):
        super(Launch_action, self).__init__(name)
        self.bt_utils = bt_utils
        self.heading = None
        self.altitude = None
        self.launch_missile = False
        self.red_team = red_team

    def update(self):
        self.heading = self.bt_utils.get_angle_to_enemy(from_red_perspective = self.red_team, offset = 0)
        self.altitude = 10e3
        self.launch_missile = True

        return pt.common.Status.RUNNING
    
    
class AirCombat_1v1(object):
    def __init__(self, conf, args, aim_dog, f16_dog, logs = None):
        # Evasive config file 
        self.conf = conf
        # transformation/scalling tools 
        self.tk = toolkit()
        self.gtk = Geo()
        
        # f16
        self.f16 = F16(conf = f16_dog, FlightGear= args['vizualize'])
        # f16 missiles 
        self.aim_block_names = list(self.conf.aim.keys())
        self.aim_block = {}
        for i in self.aim_block_names: 
            self.aim_block[i] = AIM(aim_dog, FlightGear=False, fg_out_directive=None)

        # f16r
        self.f16r = F16(conf = f16_dog, FlightGear= args['vizualize'])
        # f16r missiles
        self.aimr_block_names = list(self.conf.aimr.keys())        
        self.aimr_block = {}            
        for i in self.aimr_block_names: 
            self.aimr_block[i] = AIM(aim_dog, FlightGear=False, fg_out_directive=None)

        self.states_extra = {}
        
        # sam(surface to air missile)
        self.sam_block_names = list(self.conf.sam.keys())
        self.sam_block = {}
        for i in self.sam_block_names:
            self.sam_block[i] = AIM(aim_dog, FlightGear=False, fg_out_directive=None)
            self.states_extra[i] = None
            
        # integration time 
        if args['vizualize']:
            self.r_step = range(self.conf.general['fg_r_step'])
        else:
            self.r_step = range(self.conf.general['r_step'])

        # general configuration 
        self.env_name = self.conf.general['env_name']
        self.f16.name = self.conf.general['f16_name']
        self.f16r.name = self.conf.general['f16r_name']
        self.sim_time_sec_max = self.conf.general['sim_time_max']

        # load state holder
        self.observation_space = self.conf.states['f16_state_dim'] + 2 * self.conf.states['sam_state_dim']
        for i in self.sam_block_names:
            self.f16.state_block[i] = np.empty((1, self.conf.states['sam_state_dim']))
        self.action_space = np.empty((1,self.conf.states['act_space']))
        self.f16r_actions = np.zeros(3)
        
    def get_init_state_F16(self, rand_state=False):
        if rand_state:
            lat  = np.random.uniform(self.conf.f16_rand['lat'][0], self.conf.f16_rand['lat'][1])
            long = np.random.uniform(self.conf.f16_rand['long'][0], self.conf.f16_rand['long'][1])
            vel  = np.random.uniform(self.conf.f16_rand['vel'][0], self.conf.f16_rand['vel'][1])
            alt  = np.random.uniform(self.conf.f16_rand['alt'][0], self.conf.f16_rand['alt'][1])
            head = np.random.uniform(self.conf.f16_rand['heading'][0], self.conf.f16_rand['heading'][1])       
        
        else:
            lat = self.conf.f16['lat']
            long = self.conf.f16['long']
            alt = self.conf.f16['alt']
            vel = self.conf.f16['vel']
            head = self.conf.f16['heading']
        
        return lat, long, alt, vel, head
    
    def get_init_state_F16r(self):
        lat = self.conf.f16r['lat']
        long = self.conf.f16r['long']
        alt = self.conf.f16r['alt']
        vel = self.conf.f16r['vel']
        head = self.conf.f16r['heading']
        
        return lat, long, alt, vel, head
    
    def get_init_state_AIM(self, fdm=None, name=None, lat_tgt=None, long_tgt=None, rand_state=False):
        if fdm is not None: # 공대공(fdm, 그니까 f16과 같은 항공기 객체에서 직접 데이터를 가져옴)
            # Use fdm object to get initial state
            lat = fdm.get_lat_gc_deg()
            long = fdm.get_long_gc_deg()
            alt = fdm.get_altitude()
            vel = fdm.get_true_airspeed()
            heading = fdm.get_psi()
        else: # 지대공
            # Use configuration or random state
            if rand_state:
                lat, long, d, b = self.gtk.get_random_position_in_circle(lat0=lat_tgt,
                                                    long0=long_tgt,
                                                    d=self.conf.sam_rand[name]['distance'],
                                                    b=self.conf.sam_rand[name]['bearing'])
                alt = np.random.uniform(self.conf.sam_rand[name]['alt'][0], 
                                        self.conf.sam_rand[name]['alt'][1])
                vel = np.random.uniform(self.conf.sam_rand[name]['vel'][0], 
                                        self.conf.sam_rand[name]['vel'][1])
                heading = self.gtk.get_bearing(lat, long, lat_tgt, long_tgt)
            else:
                lat, long = self.gtk.db2latlong(lat0=lat_tgt, long0=long_tgt,
                                                d=self.conf.sam[name]['distance'], 
                                                b=self.conf.sam[name]['bearing'])
                alt = self.conf.sam[name]['alt']
                vel = self.conf.sam[name]['vel']
                heading = self.gtk.get_bearing(lat, long, lat_tgt, long_tgt)
        
        return lat, long, alt, vel, heading

    def reset_count(self):
        self.count = 0
    def reset_health(self):
        self.f16_alive = True
        self.f16r_alive = True
    def reset_reward(self):
        self.reward = None
        self.dist_min = None
        self.reward_f16_dead = 0
        self.reward_f16r_dead = 0
        self.reward_aim_hit_ground = 0
        self.reward_f16_hit_ground = 0
        self.reward_f16r_hit_ground = 0
        self.reward_sam_hit_ground = 0
        self.reward_all_lost = 0
        self.reward_max_time = 0
        
    def reset(self, rand_state_f16=False, rand_state_sam=False):
        self.reset_count()
        self.reset_health()
        self.reset_reward()
        
        # Reset F16 state (아군 항공기)
        lat0, long0, alt0, vel0, heading0 = self.get_init_state_F16(rand_state_f16)
        # deg , deg , meters, m/s, deg 
        self.f16.reset(lat0, long0, alt0, vel0, heading0)
        self.f16.state = np.zeros(self.conf.states['f16_state_dim']) # 0으로 채워진 배열로 초기화
        # 예)   self.conf.states['obs_space'] = 10이면
        #       self.f16.state = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        # Reset F16r state (적군 항공기)
        lat0, long0, alt0, vel0, heading0 = self.get_init_state_F16r()
        self.f16r.reset(lat0, long0, alt0, vel0, heading0)
        
        # Reset air-to-air missiles (공대공 미사일)
        for key in self.aim_block:
            self.aim_block[key].reset_target(None, set_active = False)
        for key in self.aimr_block:
            self.aimr_block[key].reset_target(None, set_active = False)
            
        # Reset ground-based (SAM) missiles (지대공 미사일)
        for key in self.sam_block:
            # hard reset 
            if self.sam_block[key].value_error:
                for i in self.sam_block_names:            
                        fg = False
                        fg_out = None
                        from jsb_gym.TAU.config import aim_BVRGym
                        self.sam_block[i] = AIM(aim_BVRGym, fg, fg_out)
            
            lat, long, alt, vel, heading = self.get_init_state_AIM(fdm=None, name=key, lat_tgt=lat0, long_tgt=long0, rand_state=rand_state_sam)
            self.sam_block[key].reset(lat, long, alt, vel,heading)
            self.sam_block[key].reset_target(self.f16, set_active=True)
            
        # behavior tree for the f16r
        self.BTr = BVRDog_BT(self)

        self.update_states()

        return self.f16.state, self.f16.state_block

    # 두 항공기 또는 항공기와 미사일 간의 거리 계산
    def get_distance_to_enemy(self, fdm1, fdm2, track_launch = False, scale=False, offset=None):
        
        fdm1_pos = (fdm1.get_lat_gc_deg(), fdm1.get_long_gc_deg())
        if track_launch:
            fdm2_pos = (fdm2.lat0, fdm2.long0)
        else:
            fdm2_pos = (fdm2.get_lat_gc_deg(), fdm2.get_long_gc_deg())

        dist = geodesic(fdm2_pos, fdm1_pos).meters
        
        if offset != None:
            dist += offset

        self.f16f16r_sep = dist
        # scale between -1 to 1
        if scale:
            dist = self.tk.scale_between(a= dist,\
                                         a_min= self.conf.sf['d_min'],\
                                         a_max=self.conf.sf['d_max'])
        return dist
    
    # 두 항공기 또는 항공기와 미사일 사이의 각도 계산
    def get_angle_to_firing_position(self, fdm1, fdm2, track_launch = False, scale= False, offset=None):

        fdm1_lat = fdm1.get_lat_gc_deg()
        fdm1_long = fdm1.get_long_gc_deg()

        if track_launch:
            fdm2_lat = fdm2.lat0
            fdm2_long = fdm2.long0
        else:
            fdm2_lat = fdm2.get_lat_gc_deg()
            fdm2_long = fdm2.get_long_gc_deg()

        ref_yaw = self.gtk.get_bearing(fdm1_lat, fdm1_long, fdm2_lat, fdm2_long)
        self.affp = ref_yaw
        if offset != None:
            ref_yaw += offset
        
        # range -180 to 180 deg 
        ref_yaw = self.tk.get_heading_difference(psi_ref= ref_yaw, psi_deg= fdm1.get_psi())
        
        if scale:
            # eliminate breakpoint 
            return np.sin(np.radians(ref_yaw)), np.cos(np.radians(ref_yaw))
        else:
            return ref_yaw 
    
    # 미사일 활성화 여부 확인
    def is_missile_active(self, red_missile = False):
        missile_active = False
        missile_name = None
        if red_missile:
            for key in self.aimr_block:
                if self.aimr_block[key].active == True:
                    missile_active = True
                    missile_name = key
        else:
            for key in self.aim_block:
                if self.aim_block[key].active == True:
                    missile_active = True
                    missile_name = key
        return missile_active, missile_name
    
    # 목표물과의 상대적인 위치를 NED 좌표계에서 계산
    def get_relative_position_NED(self, f16, tgt, track_launch = False, scale = False):
        lat0 = f16.get_lat_gc_deg()
        lon0 = f16.get_long_gc_deg()
        h0   = f16.get_altitude()

        if track_launch:
            # Object of interest lat, lon, h
            lat = tgt.lat0
            lon = tgt.long0
            h   = tgt.alt0
        else:
            # Object of interest lat, lon, h
            lat = tgt.get_lat_gc_deg()
            lon = tgt.get_lat_gc_deg()
            h   = tgt.get_lat_gc_deg()
        
        east, north, down = self.gtk.get_relative_unit_position_NED(lat0, lon0, h0, lat, lon, h)
        if scale:
            east = self.tk.scale_between(a= east,\
                                         a_min= self.conf.states['NE_scale'][0],\
                                         a_max= self.conf.states['NE_scale'][1])
            north = self.tk.scale_between(a= north,\
                                         a_min= self.conf.states['NE_scale'][0],\
                                         a_max= self.conf.states['NE_scale'][1])
            down = self.tk.scale_between(a= down,\
                                         a_min= self.conf.states['D_scale'][0],\
                                         a_max= self.conf.states['D_scale'][1])
        return north, east, down
    # 목표물과의 상대적인 속도를 NED 좌표계에서 계산
    def get_relative_velocity_NED(self, f16, tgt, track_launch = False, scale= False):
        vn = f16.get_v_north()
        ve = f16.get_v_east()
        vd = f16.get_v_down()

        if track_launch:
            tvn = tgt.v_n0
            tve = tgt.v_e0
            tvd = tgt.v_d0
        else:
            tvn = tgt.get_v_north()
            tve = tgt.get_v_east()
            tvd = tgt.get_v_down()

        v_north =   tvn - vn 
        v_east  =   tve - ve 
        v_down  =   tvd - vd

        if scale:
            v_north = self.tk.scale_between(a= v_north,\
                                         a_min= self.conf.states['v_NED_scale'][0],\
                                         a_max= self.conf.states['v_NED_scale'][1])
            v_east = self.tk.scale_between(a= v_east,\
                                         a_min= self.conf.states['v_NED_scale'][0],\
                                         a_max= self.conf.states['v_NED_scale'][1])
            v_down = self.tk.scale_between(a= v_down,\
                                         a_min= self.conf.states['v_NED_scale'][0],\
                                         a_max= self.conf.states['v_NED_scale'][1])
        return v_north, v_east, v_down
    
    def get_v_down(self, scale= False):
        if scale:
            return self.f16.get_v_down(scaled= True)
        else:
            return self.f16.get_v_down(scaled= False)
        
    def get_altitude(self, fdm, track_launch = False, scale = False):
        if track_launch:
            alt = fdm.alt0
        else:
            alt = fdm.get_altitude()
        if scale:
            alt = self.tk.scale_between(a= alt, a_min=self.conf.sf['alt_min'], a_max= self.conf.sf['alt_max'] )
        return alt 

    def get_velocity(self, fdm, track_launch = False, scale=False):
        if track_launch:
            vel = fdm.vel0
        else:
            vel = fdm.get_Mach()

        if scale:
            # max speed mach 2
            vel = self.tk.scale_between(a= vel, a_min= 0, a_max= self.conf.sf['mach_max'])
            return vel
        
        else:
            return vel 

    def get_psi(self, fdm, scale= False):
        if scale:
            return self.tk.scale_between(a= fdm.get_psi(), a_min=self.conf.sf['head_min'], a_max= self.conf.sf['head_max'] )
        else:
            return fdm.get_psi()
        
    def get_aim_vel0(self, aim, scale = False):
        if scale:
            # max speed mach 2
            vel = self.tk.scale_between(a= aim.vel0, 
                                        a_min= self.conf.sf['aim_vel0_min'], 
                                        a_max= self.conf.sf['aim_vel0_max'])
            return vel

    def get_aim_alt0(self, aim, scale = False):
        if scale:
            # max speed mach 2
            alt = self.tk.scale_between(a= aim.alt0, 
                                        a_min= self.conf.sf['aim_alt0_min'], 
                                        a_max= self.conf.sf['aim_alt0_max'])
            return alt
        
    def get_time_since_missile_active(self, f16, scale = False, offset = None):
        flight_time = f16.get_sim_time_sec()

        if offset != None:
            flight_time += offset
            if flight_time < 0:
                flight_time = 0

        if scale:
            flight_time = self.tk.scale_between(a=flight_time, a_min= 0,\
                                                a_max = self.conf.sf['t'])
        return flight_time
        
    def update_states(self):
        if self.conf.states['update_states_type'] == 1:
            # F16, F16r
            self.f16.state[0] = self.get_distance_to_enemy(fdm1=self.f16, fdm2=self.f16r, scale=True)
            self.f16.state[1], self.f16.state[2] = self.get_angle_to_firing_position(self.f16, self.f16r, scale=True)
            self.f16.state[3] = self.get_velocity(fdm=self.f16, scale = True)
            self.f16.state[4] = self.get_altitude(fdm=self.f16, scale = True)
            self.f16.state[5] = self.get_psi(self.f16, scale = True)
            self.f16.state[6] = self.get_velocity(fdm=self.f16r, scale = True)
            self.f16.state[7] = self.get_altitude(fdm=self.f16r, scale = True)
            # AIMr
            aim_active, aim_name = self.is_missile_active(red_missile= True)
            if aim_active:
                self.f16.state[8] = self.get_distance_to_enemy(self.f16, self.aimr_block[aim_name], track_launch= True, scale=True)
                self.f16.state[9], self.f16.state[10] = self.get_angle_to_firing_position(self.f16, self.aimr_block[aim_name], track_launch= True, scale=True)
                self.f16.state[11] = self.get_velocity(fdm=self.f16, scale = True)
                self.f16.state[12] = self.get_altitude(fdm=self.f16, scale = True)
                self.f16.state[13] = self.get_velocity(fdm=self.aimr_block[aim_name], track_launch= True, scale = True)
                self.f16.state[14] = self.get_altitude(fdm=self.aimr_block[aim_name], track_launch= True, scale = True)
            else:
                self.f16.state[8] = self.f16.state[0]
                self.f16.state[9] = self.f16.state[1]
                self.f16.state[10] = self.f16.state[2]
                self.f16.state[11] = self.f16.state[3]
                self.f16.state[12] = self.f16.state[4]
                self.f16.state[13] = self.f16.state[6]
                self.f16.state[14] = self.f16.state[7]
            # SAM
            for key in self.sam_block:
                self.f16.state_block[key][0,0] = self.get_distance_to_enemy(self.f16, self.sam_block[key], track_launch=True, scale=True)
                self.f16.state_block[key][0,1] = self.get_time_since_missile_active(self.f16, scale=True)
                self.f16.state_block[key][0,2], self.f16.state_block[key][0,3] = self.get_angle_to_firing_position(self.f16, self.sam_block[key], scale=True)
                self.f16.state_block[key][0,4] = self.get_velocity(fdm=self.f16, scale=True)
                self.f16.state_block[key][0,5] = self.get_altitude(fdm=self.f16, scale=True)
                self.f16.state_block[key][0,6] = self.get_psi(fdm=self.f16, scale=True)
                self.f16.state_block[key][0,7] = self.get_aim_vel0(aim=self.sam_block[key], scale=True)
                self.f16.state_block[key][0,8] = self.get_aim_alt0(aim=self.sam_block[key], scale=True)
                self.f16.state_block[key][0,9] = self.get_v_down(scale=True) 
                self.states_extra[key] = self.affp
        elif self.conf.states['update_states_type'] == 2:
            # F16, F16r
            north, east, down = self.get_relative_position_NED(f16=self.f16, tgt=self.f16r, scale=True)
            self.f16.state[0] = north
            self.f16.state[1] = east
            self.f16.state[2] = down

            self.f16f16r_sep = np.linalg.norm(np.array([north, east, down]))

            v_north, v_east, v_down = self.get_relative_velocity_NED(f16=self.f16, tgt=self.f16r, scale= True)
            self.f16.state[3] = v_north
            self.f16.state[4] = v_east
            self.f16.state[5] = v_down

            self.f16.state[6] = self.get_altitude(fdm = self.f16, scale = True)
            self.f16.state[7] = self.get_altitude(fdm = self.f16r, scale = True)        
            # AIMr
            aim_active, aim_name = self.is_missile_active(red_missile= True)
            if aim_active:
                north, east, down = self.get_relative_position_NED(f16=self.f16, tgt=self.aimr_block[aim_name], track_launch=True, scale=True)
                self.f16.state[8] = north
                self.f16.state[9] = east
                self.f16.state[10] = down

                v_north, v_east, v_down = self.get_relative_velocity_NED(f16=self.f16, tgt=self.aimr_block[aim_name], track_launch= True, scale= True)
                self.f16.state[11] = v_north
                self.f16.state[12] = v_east
                self.f16.state[13] = v_down

                self.f16.state[14] = self.get_altitude(fdm=self.aimr_block[aim_name], track_launch= True, scale = True)
            else:
                self.f16.state[8:14] = self.f16.state[:6]
            # SAM
            for key in self.aim_block:
                north, east, down = self.get_relative_position_NED(f16=self.f16, aim=self.aim_block[key], track_launch=True, scale=True)
                self.f16.state_block[key][0,0] = north
                self.f16.state_block[key][0,1] = east
                self.f16.state_block[key][0,2] = down
                
                self.f16.state_block[key][0,3] = self.get_time_since_missile_active(self.f16, scale=True)
                self.f16.state_block[key][0,4] = self.get_altitude(fdm=self.f16, scale = True)
                self.f16.state_block[key][0,5] = self.get_aim_alt0(aim=self.aim_block[key],scale = True)        
                
                v_north, v_east, v_down = self.get_relative_velocity_NED(f16=self.f16, aim=self.aim_block[key], scale= True)
                self.f16.state_block[key][0,6] = v_north
                self.f16.state_block[key][0,7] = v_east
                self.f16.state_block[key][0,8] = v_down
                #self.states_extra[key] = self.affp
        elif self.conf.states['update_states_type'] == 3:
            # F16, F16r
            self.f16.state[0] = self.get_distance_to_enemy(fdm1=self.f16, fdm2=self.f16r, scale=True)
            self.f16.state[1], self.f16.state[2] = self.get_angle_to_firing_position(self.f16, self.f16r, track_launch= False,scale=True)
            self.f16.state[3] = np.sin(np.radians(self.get_psi(self.f16, scale = False)))
            self.f16.state[4] = np.cos(np.radians(self.get_psi(self.f16, scale = False)))
            # AIMr
            aim_active, aim_name = self.is_missile_active(red_missile= True)
            if aim_active:
                self.f16.state[5] = self.get_distance_to_enemy(self.f16, self.aimr_block[aim_name], track_launch= True,scale=True)
                self.f16.state[6], self.f16.state[7] = self.get_angle_to_firing_position(self.f16, self.aimr_block[aim_name], track_launch= True, scale=True)
            else:
                self.f16.state[5] = self.f16.state[0]
                self.f16.state[6] = self.f16.state[1]
                self.f16.state[7] = self.f16.state[2]
            # SAM
            for key in self.aim_block:
                # missile position 
                self.f16.state_block[key][0,0] = self.get_distance_to_enemy(self.f16, self.aim_block[key], track_launch=True, scale=True)
                self.f16.state_block[key][0,1], self.f16.state_block[key][0,2] = self.get_angle_to_firing_position(self.f16, self.aim_block[key], scale=True)
                # altitude               
                self.f16.state_block[key][0,3] = self.get_altitude(fdm=self.f16, scale = True)
                # heading 
                self.f16.state_block[key][0,4] = np.sin(np.radians(self.get_psi(scale = False)))
                self.f16.state_block[key][0,5] = np.cos(np.radians(self.get_psi(scale = False)))
    
    # 시뮬레이션 종료 조건
    def is_done(self):
        all_sams_lost = all(self.sam_block[key].is_target_lost() for key in self.sam_block)
        if all_sams_lost:           
            print('all SAMs lost target')
            
        for key in self.aimr_block:
            # if red hit blue
            if self.aimr_block[key].is_target_hit():
                self.f16_alive = False
                self.reward_f16_dead = 1
                print('F16 Dead by AAM')
                return True
        for key in self.sam_block:
            if self.sam_block[key].is_target_hit():
                self.f16_alive = False
                self.reward_f16_dead = 1
                print('F16 Dead by SAM')
                return True
        
        if self.f16.get_altitude() < 1e3:
            self.f16_alive = False
            self.reward_f16_hit_ground = 1            
            print('F16 hit ground')
            return True
        
        for key in self.aim_block:
            # if blue hit red
            if self.aim_block[key].is_target_hit():
                self.f16r_alive = False
                self.reward_f16r_dead = 1
                print('F16r Dead')
                return True
        
        if (self.f16r.get_altitude() < 1e3):
            self.f16r_alive = False
            self.reward_f16r_hit_ground = 1
            print('F16r hit ground')
            return True
                
        if self.f16.get_sim_time_sec() > self.conf.general['sim_time_max']:
            self.reward_max_time = 1
            print('Max time', self.f16.get_sim_time_sec())
            return True
        else:
            return False
    
    
    # 보상 함수
    def calculate_md(self):
        for key in self.sam_block:
            dist = self.sam_block[key].position_tgt_NED_norm_min
            if self.dist_min == None:
                self.dist_min = dist
            else:
                if self.dist_min > dist:
                    self.dist_min = dist
        if self.conf.general['scale']:
            if self.dist_min > self.conf.sf['d_max_reward']:
                self.dist_min = self.conf.sf['d_max_reward']
            return round(self.tk.scale_between(self.dist_min, 0, self.conf.sf['d_max_reward']), 1)
        else:
            return self.dist_min
    def get_reward(self, is_done):
        if is_done:
            md = self.calculate_md()
            if not self.f16_alive:
                # 아군 항공기가 격추된 경우
                if self.conf.general['scale']:
                    return -1.0
                else:
                    return 0.0
            elif not self.f16r_alive:
                # 아군 항공기가 생존하고 적군 항공기가 격추된 경우
                return md + 1
            elif all(self.sam_block[key].is_target_lost() for key in self.sam_block):
                # 아군 항공기가 생존하고 모든 지대공 미사일이 목표를 잃은 경우
                return md
            else:
                return -1.0
        else:
            return 0.0
        
    def step_aim(self):
        for key in self.aim_block:
                if self.aim_block[key].active and self.aim_block[key].target_lost == False:
                    self.aim_block[key].step_evasive(name=key)

        for key in self.aimr_block:
            if self.aimr_block[key].active and self.aimr_block[key].target_lost == False:
                    #print('step evasive', key)
                    self.aimr_block[key].step_evasive(name=key)
                    #print(self.aimr_block[key].get_sim_time_sec())
    
    def get_angle_to_enemy(self, enemy_red = True, offset = 0, cache_angle = False):
        # 적 항공기와 상대적인 각도를 계산
        if enemy_red: # red가 타겟
            fdm_tgt_lat = self.f16r.get_lat_gc_deg()
            fdm_tgt_long = self.f16r.get_long_gc_deg()
            fdm_lat = self.f16.get_lat_gc_deg()
            fdm_long = self.f16.get_long_gc_deg()
            ref_yaw = self.gtk.get_bearing(fdm_lat, fdm_long, fdm_tgt_lat, fdm_tgt_long)
            ref_yaw = self.tk.get_heading_difference(psi_ref= ref_yaw + offset , psi_deg= self.f16.get_psi())       
            if cache_angle:
                self.angle_to_f16r = ref_yaw # 계산된 각도를 저장
        else: # blue가 타겟
            fdm_tgt_lat = self.f16.get_lat_gc_deg()
            fdm_tgt_long = self.f16.get_long_gc_deg()
            fdm_lat = self.f16r.get_lat_gc_deg()
            fdm_long = self.f16r.get_long_gc_deg()
            ref_yaw = self.gtk.get_bearing(fdm_lat, fdm_long, fdm_tgt_lat, fdm_tgt_long)
            ref_yaw = self.tk.get_heading_difference(psi_ref= ref_yaw + offset , psi_deg= self.f16r.get_psi())       

        return ref_yaw
    
    def f16_missile_launch(self, blue_armed = False):
        # F16이 미사일을 발사
        # blue_armed=True : blue 항공기가 무장되어 있음. 공격 가능 상태
        angle = self.get_angle_to_enemy(enemy_red=True, cache_angle=True)
        if blue_armed:    
            if any([(self.aim_block[key].is_tracking_target()) for key in self.aim_block]):
                # some are currently active 
                # 활성화된 미사일이 있으면 아무것도 하지 않는다.
                pass
            else:
                # if within firing range and firing angle  
                if abs(angle) < 35 and self.f16f16r_sep < 60e3: # 각도가 30도 이하, 거리 60km 이하
                    #print('Launch blue missiles') 
                    for key in self.aim_block:
                        print('Blue: Ready to launch, ', key , self.aim_block[key].is_ready_to_launch())
                        if self.aim_block[key].is_ready_to_launch():
                            lat, long, alt, vel, heading = self.get_init_state_AIM(fdm=self.f16)
                            self.aim_block[key].reset(lat, long, alt, vel, heading)
                            self.aim_block[key].reset_target(self.f16r, set_active=True)
                            break 
    
    def f16r_missile_launch(self, red_armed = True):
        # F16r이 미사일을 발사
        # red_armed=True: red 항공기가 무장되어 있음. 공격 가능 상태
        if self.BTr.launch_missile and red_armed:
            if any([(self.aimr_block[key].is_tracking_target()) for key in self.aimr_block]):
                    pass
            else:
                for key in self.aimr_block:                
                    if self.aimr_block[key].is_ready_to_launch():
                        lat, long, alt, vel, heading = self.get_init_state_AIM(fdm=self.f16r)
                        self.aimr_block[key].reset(lat, long, alt, vel, heading)
                        self.aimr_block[key].reset_target(self.f16, set_active = True)
                        break
    
    def print_active_missiles(self):
        for key in self.aimr_block:
            if self.aimr_block[key].active:
                print('Active: ', key, 'lost: ', self.aimr_block[key].target_lost, 'hit: ', self.aimr_block[key].target_hit )

        for key in self.aim_block:
            if self.aim_block[key].active:
                print('Active: ', key)
    
    # 환경의 상태를 한 단계 앞으로 진행시키는 함수
    # 액션을 입력으로 받아, 환경을 업데이트하고, 다음 상태, 보상, 종료 여부 등을 반환
    def step(self, action, action_type, blue_armed = False, red_armed= True):
        #if  PPO continues action -1 to 1
        for _ in self.r_step:
            #self.print_active_missiles()
            self.f16_missile_launch(blue_armed)
            self.f16r_missile_launch(red_armed)
             
            self.step_aim()
            self.f16.step_BVR(action, action_type=action_type)
            for key in self.sam_block:
                if not self.sam_block[key].is_target_lost():
                    self.sam_block[key].step_evasive(name=key)
                    
            # BT is used for the red teams aircraft 
            self.BTr.tick()
            self.f16r_actions[0] = self.tk.scale_between(a= self.tk.truncate_heading(self.BTr.heading), a_min= self.conf.sf['head_min'], a_max= self.conf.sf['head_max'])
            self.f16r_actions[1] = self.tk.scale_between(a= self.BTr.altitude, a_min= self.conf.sf['alt_min'], a_max= self.conf.sf['alt_max'])
            self.f16r_actions[2] = 0.0
            
            self.f16r.step_BVR(self.f16r_actions, action_type=action_type)
             
        done = self.is_done()
        reward = self.get_reward(done)
        
        self.update_states()
        
        if self.conf.general['rec']:
            self.logs.record(aim= self.aim_block['aim1'], tgt= self.f16)
            
        return self.f16.state, self.f16.state_block, reward, done, None