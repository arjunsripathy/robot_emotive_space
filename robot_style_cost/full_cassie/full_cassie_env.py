import numpy as np
import pyglet
import torch
import sys
sys.path.append('../')

from env import Env, Task

# Cassie Environment Specification

INIT_STATE = np.array(2*[0.] + [0.98] + 9*[0.], dtype=np.float32)
S_NORM = np.array([ 50,  50, 1.0,   1,  1.,  0.5,
                    np.radians(5), np.radians(15),
                    np.radians(180), np.radians(5),
                    np.radians(15), np.radians(30)])

class FullCassieEnv(Env):
    def __init__(self, config):
        self.num_cones = 2

    def get_task(self, att_dict = None):
        task = CassieWalkTask(self, att_dict)
        return task

    def process_waypoints(self, traj):
        # traj.states is N x 12 (x, y, z, dx, dy, dz, r, p, y, dr, dp, dy)
        states = traj.states.T
        yaw_cos, yaw_sin = torch.cos(states[8:9]), torch.sin(states[8:9])
        nstates = states / np.expand_dims(S_NORM, axis=1)
        proc_traj = torch.cat((states[:8], yaw_cos, yaw_sin, states[9:]), axis=0)
        return proc_traj.T # N x 13

    def np_task_dict(self, task_dict):
        np_task_dict = dict()
        np_task_dict['target'] = task_dict['target'].numpy()
        np_task_dict['cones'] = [c.numpy() for c in task_dict['cones']]
        return np_task_dict

    def autolabel(self, traj, fmt='np'):
        assert fmt in {'np', 'torch'}
        waypts =  traj.states
        proc_waypts = self.process_waypoints(traj)

        p_avg = torch.mean(waypts[:, 7])
        valence = -p_avg / np.radians(10)

        avg_xyz_vel = torch.mean(torch.abs(proc_waypts[:, 3:6]))
        arousal = (avg_xyz_vel - 0.3) / 0.25

        z_avg = torch.mean(waypts[:, 2])
        dominance = (z_avg - 0.85)/ 0.1
        
        feedback = torch.stack((valence, arousal, dominance), axis=0)
        feedback = torch.clip(feedback, -1, 1)

        if fmt == 'np':
            return feedback.detach().numpy()
        elif fmt == 'torch':
            return feedback


class CassieWalkTask(Task):
    def __init__(self, env, att_dict = None):
        super().__init__(env)
        self.init_state = INIT_STATE
        if (att_dict):
            self.target = att_dict['target']
            self.cones = att_dict['cones']
            if (len(self.cones) != self.env.num_cones):
                self.env.viz.adjust_num_cones(len(self.cones))
        else:
            init_xy = np.zeros([2])
            rand_pos = lambda: 3 * np.random.random([2]) - 1.5
            
            while (True):
                ca_r = 0.25
                t_r = 0.125
                co_r = 0.25
                self.target = rand_pos()
                if (np.linalg.norm(init_xy - self.target) < 4. * (ca_r + t_r)):
                    continue
                self.cones = []
                
                it = 0
                while (len(self.cones) < self.env.num_cones):
                    it += 1
                    if it > 1000:
                        break
                    
                    cone = rand_pos()
                    invalid = False
                    if (np.linalg.norm(cone - self.target) < 1.5 * (co_r + t_r) or
                        np.linalg.norm(cone - init_xy) < 1.5 * (co_r + ca_r)):
                        invalid = True
                    for other_cone in self.cones:
                        if (np.linalg.norm(cone - other_cone) < 3 * (co_r + ca_r)):
                            invalid = True
                    if invalid:
                        continue
                    self.cones.append(cone)
                self.target = torch.tensor(self.target, dtype = torch.float32)
                self.cones = [torch.tensor(cone, dtype = torch.float32) for cone in self.cones]
                if (len(self.cones) == self.env.num_cones):
                    break

    @property
    def att_dict(self):
        return {'target': self.target, 'cones': self.cones}







