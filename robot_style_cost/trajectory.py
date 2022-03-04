import torch

# Class representing an arbitrary trajectory
class Trajectory():
    def __init__(self, task, ctrls, states, dTs, seed = None):
        '''
        task: Task instance defining a task in the environment.
        ctrls: piecewise constant controls applied between waypoints ([N x CTRL_DIM])
        states: subsequent state matrix ([N x STATE_DIM])
        dTs: time deltas between waypoints
        seed: Seed used to initialize base trajectory, if generated.

        If you include init_state, N + 1 waypoints spanning sum(dTs) time.
        '''
        self.task_dict = task.att_dict
        self.init_state = task.init_state
        self.ctrls = ctrls
        self.states = states
        self.dTs = dTs
        self.seed = seed
        self.base = None

    @property
    def N(self):
        # Returns number of controls the trajectory has.
        return len(self.states)

    @property
    def time(self):
        return torch.sum(self.dTs)
    
    @property
    def base_id(self):
        # Returns string identifying the trajectory
        return f"{self.seed} {self.N} {self.task_dict}"

# Class representing a style latent label
class StyleLatentLabel():
    def __init__(self, traj, style_latent):
        '''
        traj, style: Trajectory & Style Latent
        '''
        self.traj = traj
        self.style_latent = style_latent

class StyleCostLabel():
    def __init__(self, traj, style_lang, cost):
        '''
        traj, style: Trajectory & Style Cost
        '''
        self.traj = traj
        self.style_lang = style_lang
        self.cost = cost

