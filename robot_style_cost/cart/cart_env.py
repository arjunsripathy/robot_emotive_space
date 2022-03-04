import numpy as np
import pyglet
import torch
import torch.nn.functional as F
import sys
sys.path.append('../')

from env import Env, Task
from trajectory import Trajectory

# VacuumBot Environment specification

# Visualization Constants
WIN_WIDTH = 1280
WIN_HEIGHT = 300
IMG_SCALE = 0.5
MAX_POLE_EXT = 0.7
POLE_Y = 140 * IMG_SCALE
INIT_STATE = np.array([WIN_WIDTH/2, np.pi/2] + [0.] * 5)

class Sprite():
    def __init__(self, name, anchor, scale):
        img = pyglet.image.load(f'resources/{name}.png').get_texture()
        self.anchor = anchor
        img.anchor_x = img.width * self.anchor[0]
        img.anchor_y = img.height * self.anchor[1]
        self.sprite = pyglet.sprite.Sprite(img, subpixel = True)
        self.sprite.scale = scale

class Visualizer():
    def __init__(self, env, avg_dt, display=True):
        self.env = env
        self.AVG_DT = avg_dt
        if display:
            self.window = pyglet.window.Window(WIN_WIDTH, WIN_HEIGHT)
        pyglet.gl.glClearColor(1,1,1,1)

        self.rig = Sprite('rig', anchor = [0.5, 0], scale = IMG_SCALE)
        self.rig.sprite.y = 0

        self.cart = Sprite('cart', anchor = [0.5, 0], scale = IMG_SCALE)
        self.cart.sprite.y = 0
        self.cart_w = self.cart.sprite.width
        self.cart_h = self.cart.sprite.height

        self.pole = Sprite('pole', anchor = [1 - MAX_POLE_EXT, 0.5], scale = IMG_SCALE)
        self.pole.sprite.y = POLE_Y
        self.pole_l = self.pole.sprite.width * MAX_POLE_EXT

        self.dust = Sprite('dust', anchor = [0.5, 0], scale = IMG_SCALE)
        self.dust.sprite.y = 0

        self.prev_state, self.new_state, self.cum_dt = INIT_STATE, INIT_STATE, 0
        self.show_target = True

    def save_prev_state(self):
        self.prev_state = np.array(self.env.state)
        self.cum_dt = 0

    def load_new_state(self):
        self.new_state = np.array(self.env.state)

    def update(self, dt):
        self.cum_dt += dt
        ratio = (self.cum_dt/self.AVG_DT)

        state = np.array(ratio * self.new_state + (1 - ratio) * self.prev_state)
        self.display(state)

    def display(self, state):
        self.cart.sprite.x = state[0]
        self.cart.sprite.y = state[2]
        pole_ext = self.env.pole_ext(state, fmt = 'np')
        pole_ret = self.pole_l - pole_ext
        self.pole.sprite.x = state[0] - pole_ret * np.cos(state[1])
        self.pole.sprite.y = POLE_Y + pole_ret * np.sin(state[1]) + state[2]
        self.pole.sprite.rotation = state[1] * 180 / np.pi
        self.dust.sprite.x = self.env.target
        self.rig.sprite.x = self.cart.sprite.x
        self.rig.sprite.y = max(0, self.cart.sprite.y)

        self.window.clear()
        self.cart.sprite.draw()
        self.rig.sprite.draw()
        self.pole.sprite.draw()
        self.dust.sprite.draw()


# Get constants from dummy initialization of Visualizer
dummy_viz = Visualizer(None, None, display=False)
CART_WIDTH = dummy_viz.cart_w
CART_HEIGHT = dummy_viz.cart_h
POLE_L = dummy_viz.pole_l

# State Constants
STATE_DIM = 6 #x, theta, y, dx, dtheta, dy, gravity
CTRL_DIM = 3 #ddx, ddtheta, ddy

# Claim Constants
CLAIM_THRESH = 20

# Dynamics Constants
MIN_STATE = torch.tensor([-.4 * CART_WIDTH, 1e-2, -.1 * CART_HEIGHT, 
                      -300., -10., -300., 0], dtype=torch.float32)
MAX_STATE = torch.tensor([WIN_WIDTH, np.pi, WIN_HEIGHT - CART_HEIGHT, 
                0, 0, 0, WIN_WIDTH], dtype=torch.float32) - MIN_STATE
MIN_POS, MAX_POS = MIN_STATE[:3], MAX_STATE[:3]
MIN_VEL, MAX_VEL = MIN_STATE[3:6], MAX_STATE[3:6]

GRAVITY = 0.3
FRICTION = 0.0
ANG_SPRING = 0.1
Y_SPRING = 0.0

MIN_CTRL = torch.tensor([-300., -10., -1e3], dtype=torch.float32)
MAX_CTRL = torch.tensor([300., 10., 1e3], dtype=torch.float32)

# State Invariants
MIN_GR_ANG = torch.tensor(np.arcsin(POLE_Y/POLE_L), dtype=torch.float32)

S_THRESH = -1
G_THRESH = 1
class CartEnv(Env):
    def __init__(self, config):
        self.AVG_DT = config.AVG_DT
        self.time_range_factor = config.TIME_RANGE_FACTOR
        self.viz = Visualizer(self, self.AVG_DT)

        self.state = INIT_STATE
    
    def dynamics(self, state, ctrl, dT, fmt):
        state, ctrl = self.to_torch(state, fmt), self.to_torch(ctrl, fmt)

        pos, vel, target = state[:3], state[3:6], state[6:7]
        clip_ctrl = F.relu(ctrl) * MAX_CTRL + F.relu(-ctrl) * MIN_CTRL
        adj_c2 = (clip_ctrl[2:] <= 0) * clip_ctrl[2:] + (clip_ctrl[2:] > 0 ) * clip_ctrl[2:] * (clip_ctrl[2:] + 1)
        adj_ctrl = torch.cat((clip_ctrl[:2], adj_c2), axis=0)
        if (pos[2] > G_THRESH):
            adj_ctrl[2] = 0
        clip_vel = clip(vel + adj_ctrl * dT, MIN_VEL, MAX_VEL)

        friction_decay = (1 - FRICTION) ** dT
        clip_vel[0] *= friction_decay
        if (torch.abs(pos[1] - np.pi/2) < torch.abs(MIN_GR_ANG - np.pi/2)):
            clip_vel[1] *= friction_decay

        if (pos[2] > G_THRESH):
            max_grav = (pos[2] / dT) + clip_vel[2]
            clip_vel[2] -= torch.minimum(GRAVITY * MAX_VEL[2], max_grav)
        
        if (pos[2] < S_THRESH):
            max_spring = -((pos[2] / dT) + clip_vel[2])
            clip_vel[2] += torch.minimum(Y_SPRING * MAX_VEL[2], max_spring)

        if (pos[1] - MIN_GR_ANG < -EPS):
            max_spring = -((pos[1] - MIN_GR_ANG)/ dT + clip_vel[1])
            clip_vel[1] += torch.minimum(ANG_SPRING * MAX_VEL[1], max_spring)
        elif (pos[1] - (np.pi - MIN_GR_ANG) > EPS):
            max_spring = (pos[1] - (np.pi - MIN_GR_ANG))/ dT + clip_vel[1]
            clip_vel[1] -= torch.minimum(ANG_SPRING * MAX_VEL[1], max_spring)

        clip_pos = clip(pos + dT * clip_vel, MIN_POS, MAX_POS)
        new_state = torch.cat((clip_pos, clip_vel, target), dim=0)
        return self.to_original(new_state, fmt)

    def apply_control(self, ctrl, dt):
        self.viz.save_prev_state()
        self.state = self.dynamics(self.state, ctrl, dt, fmt = 'np')
        self.viz.load_new_state()


    def get_task(self, att_dict = None):
        task = CartTask(self, att_dict)
        self.setup_for_task(task)
        return task

    def get_task_vector(self, att_dict):
        return CartTask.att_vector(att_dict)

    def constraints(self, trajectory, task):
        task_comp = task.claim_loss(trajectory.states[-1])
        ctrl_lim = torch.max(torch.abs(trajectory.ctrls)) - 1
        return torch.stack((task_comp, ctrl_lim), axis=0)

    def base_cost(self, trajectory, task):
        return torch.mean(torch.abs(trajectory.ctrls))

    def setup_for_task(self, task):
        self.state = task.init_state.numpy()
        self.target = task.target

    def process_waypoints(self, traj):
        states = traj.states
        n_states = (states - MIN_STATE.unsqueeze(0)) / \
                   (MAX_STATE - MIN_STATE).unsqueeze(0)
        return n_states

    def pole_ext(self, state, fmt = 'np'):
        if (fmt == 'np'):
            return np.minimum(POLE_L, (POLE_Y + state[2]) / np.sin(state[1]))
        elif (fmt == 'torch'):
            return torch.minimum(torch.tensor(POLE_L, dtype=torch.float32), 
                                (POLE_Y + state[2]) / torch.sin(state[1]))

    def random_init_state(self):
        # Random x, rest 0s
        state = np.array(self.state)
        state[0] = np.random.random() * (MAX_STATE[0] - MIN_STATE[0]) + MIN_STATE[0]
        state[1] = np.pi / 2
        state[2:] = 0.
        return torch.tensor(state, dtype=torch.float32)

    def autolabel(self, traj, fmt='np'):
        assert fmt in {'np', 'torch'}
        waypts =  traj.states

        y_feat = torch.mean(waypts[:, 2]) + 0.5 * torch.max(waypts[:, 2])
        valence = torch.min(y_feat/70, y_feat/9.25)

        x_vel = torch.mean(torch.abs(waypts[:, 3]))
        arousal = (x_vel - 90) / 40

        arm_feat = torch.mean(torch.abs(waypts[:, 1] - np.pi/2))
        dominance = (arm_feat - np.pi/5)/(np.pi/5)
        
        feedback = torch.stack((valence, arousal, dominance), axis=0)
        feedback = torch.clip(feedback, -1, 1)

        if fmt == 'np':
            return feedback.detach().numpy()
        elif fmt == 'torch':
            return feedback

EPS = 1e-8
class CartTask(Task):
    def __init__(self, env, att_dict = None):
        super().__init__(env)
        att_dict = dict() if att_dict is None else att_dict
        init_state = att_dict.get('init_state', env.random_init_state())
        self.init_state = init_state

        if 'target' in att_dict:
            self.target = att_dict['target']
        else:
            while (True):
                self.target = .1 * WIN_WIDTH + np.random.randint(WIN_WIDTH * .8)
                if (self.claim_loss(self.init_state).sum() > 10.):
                    break
        self.init_state[6] = self.target

    @property
    def att_dict(self):
        return {'init_state': self.init_state, 'target': self.target}
    
    def att_vector(att_dict):
        return torch.tensor([att_dict['target'] - att_dict['init_state'][0]], dtype=torch.float32)
    
    def claim_loss(self, state):
        pole_ext = self.env.pole_ext(state, fmt = 'torch')
        pole_tip_x = state[0] + pole_ext * torch.cos(state[1])
        pole_tip_y = state[2] + POLE_Y - pole_ext * torch.sin(state[1])

        target_dist = torch.abs(pole_tip_x - self.target) + torch.abs(pole_tip_y)
        return (target_dist / CLAIM_THRESH) - 1

    def try_claim(self):
        state_tens = torch.tensor(self.env.state, dtype = torch.float32)
        return torch.sum(self.claim_loss(state_tens)) < EPS and state_tens[2] < 5.

    @property
    def final_state(self):
        return torch.tensor([self.target, np.pi/2] + 4 * [0.] + [self.target], dtype = torch.float32)

    @property
    def expected_time(self):
        final_state = self.final_state
        pos_delta = (final_state - self.init_state)[:3]
        min_time_lb = torch.max(torch.abs(pos_delta) / MAX_VEL).numpy()
        return 2 * (1 + min_time_lb)

    def init_ctrls(self, N, dT):
        init_state = np.array(self.init_state)
        final_state = np.array(self.final_state)
        T = N * dT
        delta_x = (final_state - init_state)[0]
        ctrl_x_out = torch.tensor(4 * delta_x / (T ** 2), dtype=torch.float32)
        assert -MIN_CTRL[0] == MAX_CTRL[0]
        ctrl_x = (ctrl_x_out / MAX_CTRL[0]).unsqueeze(0)
        ctrl = torch.cat((ctrl_x, torch.zeros([2])), axis=0)
        if N % 2 == 0:
            ctrls = torch.cat((ctrl.repeat(int(N/2), 1), 
                              -ctrl.repeat(int(N/2), 1)), axis=0)
        else:
            ctrls = torch.cat((ctrl.repeat(int(N/2), 1),
                              torch.zeros([1, 3]),
                              -ctrl.repeat(int(N/2), 1)), axis=0)
        return ctrls



def clip(x, min_t, max_t):
    return torch.maximum(min_t, torch.minimum(x, max_t))







