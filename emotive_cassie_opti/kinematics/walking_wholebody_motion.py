import numpy as np
import pandas as pd
from scipy import interpolate
from utils.utils import *

'''
Generates whole body motion with the help of gait library
and exports trajectory specification to a csv file.
'''

class WalkingMotionGenerator:
    def __init__(self):
        self.motion_folder = "../motions/"
        gaitlib_path = "../motions/GaitLibrary.gaitlib"
        self.GaitLibrary = load_dict(gaitlib_path)
        self.__reset_ref_env()

    def __reset_ref_env(self):
        # self.curr_stanceLeg = 1 # 1 - right, -1 - left
        self.curr_stanceLeg = -1 # start from right leg
        self.curr_s = .0
        self.s_unsat_prev = .0
        self.t_prev = .0
        self.curr_ct = .0
        self.curr_time = .0

    def __update_ref_env(self, vx, vy, wh, time_in_sec):
        self.curr_HAlpha, self.curr_ct = self.__get_ref_gait(vx, vy, wh, self.curr_stanceLeg)
        s_unsat = self.s_unsat_prev + (time_in_sec - self.t_prev)*self.curr_ct
        self.curr_s = min(s_unsat, 1.0005) # walking phase, 1- end of a step
        if self.curr_s >= 1.0: # update stanceLeg
            self.curr_stanceLeg = -1*self.curr_stanceLeg
            self.curr_s = 0
            self.s_unsat_prev = 0
        else:
            self.s_unsat_prev = s_unsat
        self.t_prev = time_in_sec

    def __get_ref_gait(self, vx, vy, wh, stanceLeg):
        vx = np.clip(vx, self.GaitLibrary['Velocity'][0,:].min(), self.GaitLibrary['Velocity'][0,:].max())
        vy = np.clip(vy, self.GaitLibrary['Velocity'][1,:].min(), self.GaitLibrary['Velocity'][1,:].max())
        wh = np.clip(wh, self.GaitLibrary['Velocity'][2,:].min(), self.GaitLibrary['Velocity'][2,:].max())

        norm_vxy = np.linalg.norm([vx, vy])

        interp_point = np.array([vx, vy, wh])
        if stanceLeg == 1:
            HAlpha = interpolate.interpn((self.GaitLibrary['Velocity'][0,:],self.GaitLibrary['Velocity'][1,:],self.GaitLibrary['Velocity'][2,:]), \
                self.GaitLibrary['RightStance_HAlpha'], interp_point, method='linear')
            ct = interpolate.interpn((self.GaitLibrary['Velocity'][0,:],self.GaitLibrary['Velocity'][1,:],self.GaitLibrary['Velocity'][2,:]), \
                self.GaitLibrary['RightStance_ct'], interp_point, method='linear')
        else:
            HAlpha = interpolate.interpn((self.GaitLibrary['Velocity'][0,:],self.GaitLibrary['Velocity'][1,:],self.GaitLibrary['Velocity'][2,:]), \
                self.GaitLibrary['LeftStance_HAlpha'], interp_point, method='linear')
            ct = interpolate.interpn((self.GaitLibrary['Velocity'][0,:],self.GaitLibrary['Velocity'][1,:],self.GaitLibrary['Velocity'][2,:]), \
                self.GaitLibrary['LeftStance_ct'], interp_point, method='linear')
        
        HAlpha = np.reshape(HAlpha, (6, 10)).T

        # Style gait modification
        ct *= (norm_vxy + 0.25)

        return HAlpha, ct # gait alpha, 1/time of the step


    def __get_ref_states(self, vx, vy, wh, time_in_sec):
        # look forward next n*self.cassie_env_secs_per_env_step sec states
        s_unsat = self.curr_s
        if s_unsat >= 1.0: 
            stanceLeg = -1*self.curr_stanceLeg
        else:
            stanceLeg = self.curr_stanceLeg
        s = s_unsat%1.0
        HAlpha, ct = self.__get_ref_gait(vx, vy, wh, stanceLeg)
        joint_pos = bezier(HAlpha, s)
        joint_pos = np.concatenate([joint_pos[5:10],joint_pos[0:5]]) # flip the order, make the left leg motors first
        joint_vel = dbezier(HAlpha, s)*ct
        joint_vel = np.concatenate([joint_vel[5:10],joint_vel[0:5]]) # flip the order, make the left leg motors first
        return joint_pos.tolist(), joint_vel.tolist()  

    def generate_wholebody_motion_from_opti(self, sol_dict):
        self.motion = dict()
        self.__reset_ref_env()
        horizon = sol_dict['states'].shape[-1]
        pelvis_pose = [] # x y z roll pitch yaw
        left_leg_config = [] # 5 dof 
        right_leg_config = [] # 5 dof 
        for i in range(horizon):
            [x, y, z, roll, pitch, yaw] = sol_dict['states'][[0,1,2,6,7,8], i]
            z_rot = yaw
            x_rot = roll #roll * np.cos(yaw) + pitch * -np.sin(yaw)
            y_rot = pitch #roll * np.sin(yaw) + pitch * np.cos(yaw)
            pelvis_pose += [[x,y,z,x_rot,y_rot,z_rot]]
            [vx, vy, wh] = sol_dict['states'][[3,4,2], i]
            time_in_sec = sol_dict['time'][i]
            mpos, _ = self.__get_ref_states(vx, vy, wh, time_in_sec)
            mpos[0] -= roll
            mpos[5] -= roll
            mpos[2] += pitch
            mpos[7] += pitch
            self.__update_ref_env(vx, vy, wh, time_in_sec)
            left_leg_config += [mpos[:5]]
            right_leg_config += [mpos[5:]]

        self.motion['Pelvis'] = np.array(pelvis_pose).T
        self.motion['Left_Leg'] = np.array(left_leg_config).T
        self.motion['Right_Leg'] = np.array(right_leg_config).T
        self.motion['Time'] = sol_dict['time']
        self.motion['Horizon'] = horizon


    def export_motion_to_csv(self, file_path):
        data = np.zeros((21,self.motion['Horizon']))
        line = 0
        data[line,:] = self.motion['Time']
        line += 1
        for i in range(6):
            data[line,:] = self.motion['Pelvis'][i,:]
            line += 1 
        for i in range(4):
            data[line,:] = self.motion['Left_Leg'][i,:]
            line += 1
        data[line,:] = np.zeros((1,self.motion['Horizon']))
        line += 1
        data[line,:] = np.radians(13)-self.motion['Left_Leg'][3,:]
        line += 1
        data[line,:] = self.motion['Left_Leg'][4,:]
        line += 1
        for i in range(4):
            data[line,:] = self.motion['Right_Leg'][i,:]
            line += 1
        data[line,:] = np.zeros((1,self.motion['Horizon']))
        line += 1
        data[line,:] = np.radians(13)-self.motion['Right_Leg'][3,:]
        line += 1
        data[line,:] = self.motion['Right_Leg'][4,:]
        pd.DataFrame(data).to_csv(file_path, header=None, index=False)
        print("file ", file_path, " is saved")
