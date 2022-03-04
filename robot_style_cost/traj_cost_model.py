import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import os
import pickle

# Neural network style cost model

EPS = 1e-5
class CostModel(nn.Module):
    def __init__(self, env, base_cache, config):
        '''
        env: an Env, the environment for which we want to encode trajectories.
        config: StyleNet configuration

        Initializes style neural network(s).
        '''
        super().__init__()
        self.env = env
        self.style_dim = config.STYLE_LATENT_DIM
        self.dist_metric = config.CM_DIST_METRIC
        self.data_fp = config.DATA_FP
        self.method = config.METHOD
        
        if self.method == 'OURS':
            self.traj_model = TrajEncoder(env, base_cache, config)
        elif self.method == 'ALLAN':
            self.traj_models = nn.ModuleDict()
            model_keys = config.EVAL_WORDS
            print(f"ALLAN MODEL {len(model_keys)} Emotions")
            for key in model_keys:
                self.traj_models[key] = TrajEncoder(env, base_cache, config)
        if self.method in {'OURS', 'ALLAN'}:
            self.opt = torch.optim.Adam(self.parameters(), lr = config.TRAIN_LR)

    def get_style_cost_fn(self, target_style, coef):
        '''
        target_style: Target Style (Latent for OURS, Lang for ALLAN)
        coef: scalar coefficient

        returns traj cost fn encouraging projection of target_style_latent
        '''

        if self.method in {'OURS', 'ORACLE'}:
            if self.dist_metric == 'COS':
                dist_fn = lambda x, y: -torch.mean(x * y)
            elif self.dist_metric == 'SQD':
                dist_fn = lambda x, y: torch.mean((x - y) ** 2)
            target_style_latent = target_style.unsqueeze(0)

            def cost_fn(traj, verbose=False):
                traj_bb = self.traj_model(traj)
                # Minimize distance between discriminator output and target style
                return coef * dist_fn(traj_bb, target_style) 

        elif self.method == 'ALLAN':
            def cost_fn(traj, verbose=False):
                traj_bb = self.traj_models[target_style](traj)
                # Directly use specific emotion model output for cost computation
                return coef * torch.mean(traj_bb ** 2)

        return cost_fn

    def regularizer(self):
        # Get model regularizer
        if self.method == 'OURS':
            return self.traj_model.regularizer()
        elif self.method == 'ALLAN':
            regs = []
            for traj_model in self.traj_models.values():
                regs.append(traj_model.regularizer())
            return torch.mean(torch.stack(regs))

    def save_params(self):
        save_fp = f'{self.data_fp}/cost_model.pt'
        torch.save(self.state_dict(), save_fp)

    def load_params(self):
        save_fp = f'{self.data_fp}/cost_model.pt'
        if os.path.exists(save_fp):
            state_dict = torch.load(save_fp)
            self.load_state_dict(state_dict)
            print(f"Loaded params for traj model.")
        else:
            print(f"Couldn't find params for traj model.")

    def train(self, loss):
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

class TrajEncoder(nn.Module):
    def __init__(self, env, base_cache, config):
        # Style latent discriminator
        super().__init__()
        self.env = env
        self.base_cache = base_cache
        self.enc_mode = config.CM_ENC_MODE
        self.method = config.METHOD
        assert self.enc_mode in {'waypt', 'traj'}
        ACTIVATION_FNS = {'relu': nn.ReLU(),
                          'tanh': nn.Tanh(),
                          'leaky_relu': nn.LeakyReLU(),
                          'elu': nn.ELU(),
                          'identity': nn.Identity()}
        self.act_fn = ACTIVATION_FNS[config.CM_ACTIVATION]

        INP_DIM = config.STATE_DIM + config.TIME_DIM
        W_HDs = [INP_DIM] + config.CM_WAYPT_HID_DIMS
        T_INP_DIM = 2 * W_HDs[-1] if self.enc_mode == 'traj' else W_HDs[-1]
        T_HDs = [T_INP_DIM] + config.CM_TRAJ_HID_DIMS

        # Processed waypoint -> waypoint embedding
        waypt_layers = []
        for i in range(len(W_HDs) - 1):
            waypt_layers.append(nn.Linear(W_HDs[i], W_HDs[i+1]))
            waypt_layers.append(self.act_fn)
        self.waypt_net = nn.Sequential(*waypt_layers)

        # Optional pooling

        # Trajectory embedding -> style latent
        out_layers = []
        for i in range(len(T_HDs) - 1):
            out_layers.append(nn.Linear(T_HDs[i], T_HDs[i+1]))
            out_layers.append(self.act_fn)
        if self.method == 'OURS':
            out_layers += [nn.Linear(T_HDs[-1], config.STYLE_LATENT_DIM), nn.Tanh()]
        elif self.method == 'ALLAN':
            out_layers.pop()

        self.out_net = nn.Sequential(*out_layers)

    def process_traj(self, traj):
        # Returns normalized and augmented waypoints
        waypts = self.env.process_waypoints(traj)
        N = waypts.shape[0]
        dTs = torch.reshape(traj.dTs, [-1, 1])
        aug_waypts = torch.cat((waypts, dTs), dim = 1)
        return aug_waypts

    def forward(self, traj, verbose=False):
        # Discriminates style latent from trajectory
        traj_proc = self.process_traj(traj) # Preprocess trajectory
        traj_inp = self.waypt_net(traj_proc) # NN on each waypoint independently
        if self.enc_mode == 'traj':
            # Pool waypoints by concatenating softmax and average pool embeddings
            softmax_feats = torch.log(torch.sum(torch.exp(traj_inp), dim=0, keepdim=True))
            mean_feats = torch.mean(traj_inp, dim=0, keepdim=True)
            traj_inp = torch.cat((softmax_feats, mean_feats), dim=1)
        return self.out_net(traj_inp) # NN on trajectory embedding

    def regularizer(self):
        # L1 Regularizer
        l1_penalty = 0
        for name, param in self.named_parameters():
            if 'bias' not in name:
                l1_penalty += torch.sum(torch.abs(param))
        return l1_penalty








