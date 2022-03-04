import torch
from torch import nn
import numpy as np
import os
import pickle
import time

from trajectory import Trajectory
from base_cache import BaseCache

# General pytorch trajectory optimizer.

EPS = 1e-8
class TrajectoryOptimizer():
    def __init__(self, env, config, silent):
        '''
        env: an instance of the Env class for which we'd to optimize trajectories
        config: StyleNetConfiguration
        '''
        self.silent = silent
        self.env = env
        self.avg_dT = config.AVG_DT # Average time delta between waypoints
        self.max_iters = config.TO_MAX_ITERS # Maximum number of iterations to optimize
        self.check_every = config.TO_CHECK_EVERY # Check for early stopping every _ iters
        self.stop_early = config.TO_STOP_EARLY # Stop early if error reduction not this much
        self.lm = config.LM # Lagrange Multipliers
        self.lr = config.TO_LR # Learning Rate
        self.sig_lower = config.SIG_LOWER #Minimum reduction we consider signficant to save.
        self.lr_decay = config.LR_DECAY # Learning Rate Decay when unsuccessful.
        self.ckpt_every = config.TO_CKPT_EVERY #Check for checkpoint every # of iters.
        self.time_range_factor = config.TIME_RANGE_FACTOR #Maximum factor from avg dT
        self.max_sc_stop = config.TO_MAX_SC_STOP #Don't stop early if not below this style cost

        self.base_cache = BaseCache(self, config)

    def get_base_problem(self, traj):
        # Get base optimization problem for a trajectory
        task = self.env.get_task(traj.task_dict)
        base_problem = TrajOptProblem(self, 'opt_base', task, None, traj.N)
        return base_problem

    def plan(self, task, style_cost_fn = None, N = None, verbose = True):
        '''
        task: Task instance in the environment
        style_cost_fn: Function mapping waypoints to style cost.
        N: Manually provided number of ctrls
        verbose: enable/disable print statements

        Returns an optimized trajectory for the given style_cost_fn.
        '''
        if (verbose and not self.silent):
            print("Planning")

        N = N if N else int(1 + (task.expected_time // self.avg_dT))

        base_problem = TrajOptProblem(self, 'opt_base', task, None, N)
        base_sol = self.base_cache.get_base_sol(base_problem.traj)

        if (style_cost_fn is None):
            return base_sol

        problem = TrajOptProblem(self, 'opt_style', task, style_cost_fn, N, 
                                 init_traj = base_sol)
        
        return self.optimize_problem(problem, run_type='opt_style', verbose=verbose)

    def optimize_problem(self, problem, run_type, verbose = True, aux = None):
        '''
        problem: Problem to optimize
        run_type: Type of optimization ('opt', 'est', etc.)
        verbose: enable/disable print statements
        aux: If provided this function is called every iteration with it as an argument

        Optimizes problem and returns solution along with final full cost.
        '''
        prev_cost = problem.full_cost(optimize = False)[0] 
        it = 0
        best, min_cost = None, float('inf')
        for i in range(self.max_iters[run_type] // self.check_every[run_type]):
            for j in range(self.check_every[run_type]):
                if (j % self.ckpt_every == 0):
                    ckpt = problem.solution(copy = True)

                it_ratio = it / self.max_iters[run_type]
                full_cost, base_cost, style_cost = problem.full_cost(optimize = True, it_ratio=it_ratio)
                if (j % self.ckpt_every == 0 and full_cost < self.sig_lower * min_cost):
                    best = ckpt, (full_cost, base_cost, style_cost)
                    min_cost = full_cost

                it += 1
                if (verbose and not self.silent):
                    print(f"Iter {it}: {full_cost:.3f} (BC {base_cost:.3f}, SC {style_cost:.3f})", end = "\r")

                if (aux):
                    aux(it)

            if (min_cost >= self.stop_early[run_type] * prev_cost):
                if (best[1][2] < self.max_sc_stop):
                    break
                else:
                    problem.initialize_from_ctrls(best[0].ctrls, drop_lr = True)
            prev_cost = min_cost

        if (verbose and not self.silent):
            print(f"Best Iteration {best[1][0]:.3f} (BC {best[1][1]:.3f}, SC {best[1][2]:.3f})")

        return best[0]



# Contains information specifying traj opt problem & solution
class TrajOptProblem():
    def __init__(self, traj_opt, run_type, task, style_cost_fn, N, 
                 init_traj = None, inc_base_cost = True):
        '''
        traj_opt: Reference to TrajectoryOptimizer that created this.
        run_type: 'opt_style' or 'opt_base'
        task: Task we're trying to accomplish
        style_cost_fn: Style cost function
        N: Number of controls
        dT: Duration we apply each control
        inc_base_cost: If False don't include base cost

        Sets up parameters to solve trajectory optimization problem.
        '''
        # Problem specification
        self.env = task.env
        self.run_type = run_type
        self.task = task
        self.init_state = task.init_state
        self.style_cost_fn = style_cost_fn
        self.N = N
        self.avg_dT = torch.tensor(traj_opt.avg_dT, dtype=torch.float32)
        self.min_log_dT = torch.log(self.avg_dT / traj_opt.time_range_factor)
        self.max_log_dT = torch.log(self.avg_dT * traj_opt.time_range_factor)
        self.inc_base_cost = inc_base_cost
        self.lm = torch.tensor(traj_opt.lm[self.run_type], dtype=torch.float32)
        self.lr = traj_opt.lr[self.run_type]
        self.lr_decay = traj_opt.lr_decay
        
        if init_traj:
            init_ctrls = torch.clone(init_traj.ctrls)
        else:
            init_dT = self.avg_dT.detach().numpy()
            init_ctrls = self.task.init_ctrls(self.N, init_dT)
        self.initialize_from_ctrls(init_ctrls)


    def initialize_from_ctrls(self, init_ctrls, drop_lr = False):
        '''
        init_ctrls: initial control tensor
        
        Initialize controls parameter and trajectory solution as well as
        lagrange multipliers.
        '''
        self.N = len(init_ctrls)
        self.ctrl_scalars = nn.Parameter(init_ctrls + 1e-4 * torch.randn_like(init_ctrls))
        self.dT_logit = nn.Parameter(torch.zeros(1))
        self.traj = Trajectory(self.task, None, None, None)
        self.update_waypoints()

        if (drop_lr):
            self.lr *= self.lr_decay

        params = [self.dT_logit, self.ctrl_scalars]
        self.optimizer = torch.optim.Adam(params, lr = self.lr)
        self.constr_zeros = torch.zeros_like(self.lm)

    def compute_dT(self):
        # Compute dTs from logit representation
        scalar = torch.sigmoid(self.dT_logit)
        log_dT = self.min_log_dT + scalar * (self.max_log_dT - self.min_log_dT)
        return torch.exp(log_dT)

    def update_waypoints(self):
        # Update waypoints based on dTs and ctrl scalars
        dT = self.compute_dT()
        ctrls = self.ctrl_scalars * self.avg_dT / dT
        dTs = dT.repeat(self.N)
        self.traj.dTs = dTs
        self.traj.ctrls = ctrls
        self.traj.states =  self.env.waypoints(self.init_state, ctrls, dTs, 'torch')

    def full_cost(self, optimize = False, it_ratio=0):
        '''
        optimize: If True take an optimization step to reduce.

        Returns the enviornment base cost & constraint combined with the style cost if any,
        as well as the style cost on it's own.
        '''
        base_cost = self.env.base_cost(self.traj, self.task) if self.inc_base_cost else 0.
        style_cost = self.style_cost_fn(self.traj) if self.style_cost_fn else 0.
        violations = self.env.constraints(self.traj, self.task)
        constraint_cost = torch.sum(self.lm * torch.maximum(violations, torch.zeros(1)))

        full_cost = base_cost + style_cost + constraint_cost

        if (optimize):
            self.optimizer.zero_grad()
            full_cost.backward()
            self.optimizer.step()
            self.update_waypoints()

        base_cost = base_cost.detach().numpy() if self.inc_base_cost else 0.
        style_cost = style_cost.detach().numpy() if self.style_cost_fn else 0.
        return full_cost.detach().numpy(), base_cost, style_cost


    def solution(self, copy = False):
        '''
        copy: If true we return a cloned version of the trajectory.

        Detaches parameters and returns solution trajectory.
        '''
        if (copy):
            waypts = self.env.waypoints(self.init_state, self.traj.ctrls, 
                                   self.traj.dTs, 'torch').detach()
            return Trajectory(self.task, torch.clone(self.traj.ctrls.detach()), 
                              waypts, torch.clone(self.traj.dTs.detach()))
        else:
            waypts = self.env.waypoints(self.init_state, self.traj.ctrls, 
                                  self.traj.dTs, 'torch').detach()
            self.traj.states = waypts
            self.traj.ctrls = self.traj.ctrls.detach()
            self.traj.dTs = self.traj.dTs.detach()
            return self.traj




        