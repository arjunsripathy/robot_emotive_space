import torch
import numpy as np
import pyglet
import os

# Abstract class for style cost experiment environments
class Env():
    def dynamics(self, state, ctrl, dT, fmt = 'np'):
        '''
        state: state vector
        ctrl: control vector
        dT: duration of time
        fmt: 'np' or 'torch', (are state, ctrl, dt numpy or torch?)

        Returns the new state after applying ctrl from state for dT.
        '''
        raise NotImplementedError

    def waypoints(self, state, ctrls, dTs, fmt = 'np'):
        '''
        state: initial state tensor
        ctrls: control matrix ([N x CTRL_DIM])
        dTs: time delta between waypoints
        fmt: 'np' or 'torch'

        Returns the resulting waypoints from starting at state, and applying
        ctrls, where each control is continuously applied for the dT seconds.
        The initial state is left out of the returned waypoints so the number of
        waypoints matches the number of controls.
        '''
        N = len(ctrls)
        states = []
        for i in range(N):
            prev_state = states[-1] if len(states) > 0 else state
            states.append(self.dynamics(prev_state, ctrls[i], dTs[i], fmt))
        if (fmt == 'torch'):
            return torch.stack(states)
        else:
            return np.stack(states)

    def visualize_trajectory(self, trajectory, save_frames=None):
        '''
        trajectory: Trajectory instance

        Visualize the provided trajectory in the environment. If save_frames
        is provided then we same snapshots of the trajectory at a frames
        per second rate equal to the value given in save_frames.
        '''
        N = trajectory.N
        task = self.get_task(trajectory.task_dict)
        self.setup_for_task(task)

        save_inds = []
        if (save_frames):
            for img_fp in os.listdir(self.frames_fp):
                print(f"REMOVING {img_fp}")
                os.remove(f"{self.frames_fp}/{img_fp}")

            save_inds, cur_time = [], 0
            for i in range(N):
                if cur_time >= len(save_inds) / save_frames:
                    save_inds.append(i)
                cur_time += trajectory.dTs[i].numpy()


        first_step, Ti, leftover, next_save = True, 0, 0, 0
        def step(dt):
            nonlocal first_step, Ti, leftover, next_save
            if (first_step):
                first_step = False
                return

            leftover += dt
            while Ti < N and leftover >= trajectory.dTs[Ti]:
                leftover -= trajectory.dTs[Ti]
                Ti += 1

            fp = f"{self.frames_fp}/{Ti}.png"
            if (Ti == N):
                if save_frames is not None:
                    pyglet.image.get_buffer_manager().get_color_buffer().save(fp)
                pyglet.clock.unschedule(step)
                pyglet.app.exit()
            else:
                if (Ti == 0):
                    prev_waypt = trajectory.init_state
                else:
                    prev_waypt = trajectory.states[Ti - 1] 
                next_waypt = trajectory.states[Ti]
                ratio = leftover / trajectory.dTs[Ti]
                self.state = np.array(ratio * next_waypt + (1 - ratio) * prev_waypt)
                self.viz.display(self.state)

                if (save_inds and Ti >= save_inds[0]):
                    save_inds.pop(0)
                    pyglet.image.get_buffer_manager().get_color_buffer().save(fp)

        pyglet.clock.schedule(step)
        pyglet.app.run()

    def get_task(self, att_dict = None):
        '''
        att_dict: attribute dictionary for Task in the environment.

        Returns a Task instance which contains all necessary environmental information
        to optimize a trajectory which complete the objective. Setups up environment.
        If att_dict not provided then an arbitrary task is returned.
        '''
        raise NotImplementedError

    def constraints(self, trajectory, task):
        '''
        trajectory: Trajectory instance
        task: Task instance

        Returns a tensor of values each of which we'd like to be less than or equal to 0.
        These represent the kinematics, dynamics, and goal constraints of the trajectory.
        '''
        raise NotImplementedError

    def base_cost(self, trajectory, task):
        '''
        trajectory: Trajectory instance
        task: Task instance

        Returns a scalar tensor which represents the base cost for the suggested trajectory,
        ignoring any constraints.  For example the smoothness of the trajectory may be involved.
        '''
        raise NotImplementedError
    
    def setup_for_task(self, task):
        '''
        task: a Task instance

        Setup environment to attempt the provided task.
        '''
        raise NotImplementedError

    def process_waypoints(self, traj):
        '''
        traj: a Trajectory instance

        Return waypoints processed & normalized for encoding.
        '''
        raise NotImplementedError

    def to_torch(self, value, fmt):
        '''
        value: np array or tensor
        fmt: 'np' or 'torch'

        Return tensor version of value
        '''
        return value if fmt == 'torch' else torch.tensor(value, dtype=torch.float32)

    def to_original(self, value, fmt):
        '''
        value: tensor
        fmt: 'np' or 'torch'

        Return value in format fmt
        '''
        return value if fmt == 'torch' else np.array(value, dtype=np.float32)


# Abstract class for tasks in an environment
class Task():
    def __init__(self, env):
        '''
        env: Env instance

        Initializes task from init_state in env. 
        '''
        self.env = env

    @property
    def expected_time(self):
        #Returns an expected time value.
        raise NotImplementedError

    def init_ctrls(self, N, dT, seed = 0):
        '''
        N: number of controls to be applied
        dT: each control will be applied for dT seconds
        seed: integer specifying random seed

        Returns a set of N controls for completing the task when applied at equally
        spaced intervals of dT seconds. Returned controls should be a [N x CTRL_DIM] 
        tensor.  Repeated calls with different seeds yields varying initial controls.
        '''
        raise NotImplementedError

    @property
    def att_dict(self):
        # Returns a dictionary specifying attributes necessary to recreate task.
        raise NotImplementedError

    def att_vector(att_dict):
        # Returns a vector representing attributes of the task.
        raise NotImplementedError



