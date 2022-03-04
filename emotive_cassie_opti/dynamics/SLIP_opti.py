import casadi as ca
import sys 
sys.path.append('..')
from SLIP_model import SLIP
import numpy as np
import time
from utils.draw import CassieDrawer

from kinematics.walking_wholebody_motion import WalkingMotionGenerator
from kinematics.style_cost_gen import get_cost_fn
import pickle
from boxsdk import DevelopmentClient
import os

'''
Main entry point for optimizing trajectories with Cassie.
Running this script will launch a job that continuously polls
for trajectory optimization "orders". Once an order, containing
a task specification and a desired style, is received it computes 
the optimal trajectory with CasADI and returns the result. If the
process generating orders is on a different machine CLOUD file transfer
is supported using Box; however, one will have to provide their own Box
folders for this purpose. If the processes are running on the same machine,
using LFSYS (local file system) will provide improved performance.
'''

POLL_FREQ = {'CLOUD': 5, 'LFSYS': 1}
while True:
    TRANS = input("TRANSPORT: ")
    if TRANS in POLL_FREQ:
        break
POLL_FREQ = POLL_FREQ[TRANS]

class SLIPOpti:
    def __init__(self):
        self.opti = ca.Opti()

        self.slip = SLIP(bound_path='../model/cmd_bound_hzd_convex_shrink.csv')
        self.dynamics_func = self.slip.singleSupportDynamics()
        self.bound_set = self.slip.getBoundSet()
        _, self.set_num = np.shape(self.bound_set)

        # state
        # [x y z dx dy dz roll pitch yaw droll dpitch dyaw]
        # input
        # [ux uy uz uroll upitch uyaw]
        self.input_num = 6
        self.state_num = 12
        
        self.avg_dt = 0.05
        self.style_dt = 0.25
        self.grid_per_phase = 6

        self.z_optimal = 0.98
        self.safe_range_obs = 0.05
        self.safe_range_height = 0.0
        self.robot_radius = 0.4

        self.Xp_init = 0.001
        self.Yp_init = 0.001
        self.Xp = 0.001
        self.Yp = 0.001

        self.w_state = np.array(6*[0] + 3*[0.5] + 3*[0.25] + 6*[1.])  # x dx ddx
        self.w_allignment = 1e3
        self.w_local_vy = 0.1
        self.w_height = 0.
        self.w_slack_convex = 1e5
        self.w_time = 1e3
        self.w_slack_final_state = np.array(3*[1e6])
        self.w_slack_final_statedot = np.array(6*[1e4])
        self.w_slack_obs = 1e6

        self.ipopt_option = {"ipopt.max_cpu_time": 10}

    def setup_grid(self, init_time):
        self.num_phases = round(init_time/(self.grid_per_phase * self.avg_dt))
        self.grid_num = int(self.num_phases * self.grid_per_phase)

    def reset(self, initState, finalState, init_sol_dict):
        self.opti = ca.Opti()
        self.opti.solver("ipopt", self.ipopt_option)
        self.Xp = 0.001
        self.Yp = 0.001     
        self.total_cost = 0

        self.time = self.opti.variable(1)
        min_time = 0.5 + 1. * np.linalg.norm(finalState[:2] - initState[:2])
        max_time = min_time * 4
        init_time = min_time * 2
        self.style_N = int(init_time / self.style_dt)
        self.setup_grid(init_time)
        if (init_sol_dict):
            init_time = init_sol_dict['total_time']
        print(f"Init Time: {init_time:.2f}")
        self.opti.set_initial(self.time, init_time)

        self.states = self.opti.variable(self.state_num, self.grid_num + 1)
        self.inputs = self.opti.variable(self.input_num, self.grid_num)
        # self.footholders = self.opti.variable(2, self.phase_num)

        self.opti.subject_to(self.time >= min_time)
        self.opti.subject_to(self.time <= max_time)

        self.slack_convex = self.opti.variable(self.grid_num)
        self.opti.subject_to(self.slack_convex[:] >= 0)

        self.w_xyvel = self.opti.variable(self.set_num, self.grid_num)
        self.opti.subject_to(self.w_xyvel[:] >= 0)
        # self.opti.subject_to(ca.sum1(self.w_xyvel) == np.ones((1,self.total_grid))+self.slack_convex.T)

        self.slack_final = self.opti.variable(self.state_num)

    def __addNodeLocalVyCost(self, x):
        _, vy_local = self._global2local(x[3], x[4], x[8])
        self.node_local_vy_cost = self.w_local_vy * (vy_local ** 2)
        self.total_cost += self.node_local_vy_cost

    def __addNodeVelCost(self, x):
        self.node_vel_cost = ca.mtimes(self.w_state[6:12].reshape((1, 6)), x[[3,4,5,9,10,11]] ** 2)
        self.total_cost += self.node_vel_cost

    def __addNodeAllignmentCost(self, x):
        dx, dy, yaw = x[3], x[4], x[8]
        heading = dx * ca.cos(yaw) + dy * ca.sin(yaw)
        ortho = dx * -ca.sin(yaw) + dy * ca.cos(yaw)
        deviance = ortho ** 2 + ca.fmax(-heading, 0.) ** 2
        self.total_cost += self.w_allignment * deviance

    def __addNodeAccCost(self, dx):
        self.node_acc_cost = ca.mtimes(self.w_state[12:18].reshape((1, 6)), dx[[3,4,5,9,10,11]] ** 2)
        self.total_cost += self.node_acc_cost

    def __addNodeHeightCost(self, x):
        self.node_height_cost = self.w_height ** ((x[2] - self.z_optimal) ** 2)
        self.total_cost += self.node_height_cost

    def __addSlackConvexCost(self, slack_convex):
        self.node_slack_convex_cost = self.w_slack_convex * ca.sum1(ca.vec(slack_convex))
        self.total_cost += self.node_slack_convex_cost

    def __addSlackFinalCost(self, slack_final):
        self.node_slack_final_cost = ca.mtimes(self.w_slack_final_state.reshape((1, 3)), slack_final[[0,1,6]] ** 2)
        # self.opti.subject_to(slack_final[3,4,5,7] == 0)
        self.node_slack_final_cost += ca.mtimes(self.w_slack_final_statedot.reshape((1, 6)), slack_final[[3,4,5,9,10,11]] ** 2)
        self.total_cost += self.node_slack_final_cost

    def __addTimeCost(self):
        self.total_cost += self.w_time * self.time

    def __addSlackObsCost(self, slack_obs):
        self.node_slack_obs_cost = self.w_slack_obs * ca.sum1(ca.vec(slack_obs**2))
        self.total_cost += self.node_slack_obs_cost

    def get_style_traj(self, states, time):
        num_states = states.shape[1]
        indices = []
        for i in range(self.style_N):
            indices.append(int((num_states-1) * i/(self.style_N-1)))
        dTs = time * np.ones([1, self.style_N])/self.style_N
        return states[:, indices], dTs

    def __addStyleCost(self, states, time):
        style_traj = self.get_style_traj(states, time)
        self.style_cost = self.style_cost_fn(style_traj)
        self.total_cost += self.style_cost

    # kinematics transform is hardcoded
    # it can only do yaw transform in xy plane
    def _global2local(self, xg, yg, yaw):
        # NOTE: yaw is first, no roll
        # rotate yaw
        xl = xg * ca.cos(yaw) + yg * ca.sin(yaw)
        yl = -xg * ca.sin(yaw) + yg * ca.cos(yaw)
        return xl, yl

    def _local2global(self, xl, yl, yaw):
        # rotate yaw
        xg = xl * ca.cos(yaw) - yl * ca.sin(yaw)
        yg = xl * ca.sin(yaw) + yl * ca.cos(yaw)
        return xg, yg

    # NOTE: new function
    def __get_time_axis(self, time):
        curr_time = 0.0
        time_list = [curr_time]
        deltaT = time / (self.grid_num-1)
        for _ in range(self.grid_num):
            curr_time += deltaT
            time_list += [curr_time]
        return time_list

    @property
    def deltaT(self):
        return self.time / self.grid_num
    
    def solve(self, initState, finalState, obstacles, init_sol_dict = None):
        self.reset(initState, finalState, init_sol_dict)

        self.opti.subject_to(self.states[:, 0] == initState)

        last_statedot = np.zeros([12], dtype=np.float32)
        last_statedot[:3] += initState[3:6]
        last_statedot[6:9] += initState[9:12]
        k = 0
        for i in range(self.grid_num):
            Xk = self.states[:, k]
            Uk = self.inputs[:, k]

            self.opti.subject_to(self.slip.input_bound_lb <= Uk)
            self.opti.subject_to(Uk <= self.slip.input_bound_ub)
            self.opti.set_initial(Uk, 0)

            Xk_next = self.states[:, k + 1]
            vx_local, vy_local = self._global2local(Xk_next[3], Xk_next[4], Xk_next[8])
            self.opti.subject_to(
                ca.vertcat(vx_local, vy_local, Xk_next[2]) == ca.mtimes(self.bound_set, self.w_xyvel[:, k])
            )
            self.opti.subject_to(ca.sum1(self.w_xyvel[:, k]) == 1 + self.slack_convex[k])

            self.opti.subject_to(self.slip.state_bound_lb <= Xk_next)
            self.opti.subject_to(Xk_next <= self.slip.state_bound_ub)

            LL_cie = (
                (Xk_next[0] - self.Xp) ** 2 + (Xk_next[1] - self.Yp) ** 2 + Xk_next[2] ** 2 - self.slip.l0_ ** 2
            )
            self.opti.subject_to(LL_cie <= 0)

            statedot = self.dynamics_func(Xk, Uk, ca.vertcat(self.Xp, self.Yp))
            state_p = Xk + (statedot + last_statedot) * self.deltaT / 2.0
            self.opti.subject_to(Xk_next == state_p)

            self.__addNodeAccCost(statedot)
            self.__addNodeVelCost(Xk_next)
            self.__addNodeLocalVyCost(Xk_next)
            self.__addNodeHeightCost(Xk_next)
            self.__addNodeAllignmentCost(Xk_next)
            
            if i % self.grid_per_phase == (self.grid_per_phase - 1):
                vx_local, vy_local = self._global2local(Xk[3], Xk[4], Xk[8])
                leg_length = self.slip.touchDownLength(vx_local, Xk[2])
                leg_pitch = self.slip.touchDownAngle(vx_local, Xk[2])
                leg_abduction = self.slip.touchDownAbduction(vy_local, Xk[2])
                # leg_length = self.slip.touchDownLength(Xk[3],Xk[2])
                # leg_pitch = self.slip.touchDownAngle(Xk[3],Xk[2])
                # leg_abduction = self.slip.touchDownAbduction(Xk[4],Xk[2])
                step_len_local = (leg_length + 0.1) * ca.cos(leg_pitch) * ca.cos(leg_abduction)
                step_width_local = (leg_length + 0.1) * ca.cos(leg_pitch) * ca.sin(leg_abduction)
                step_len_global, step_width_global = self._local2global(step_len_local, step_width_local, Xk[6])
                self.Xp = Xk[0] + step_len_global
                self.Yp = Xk[1] + step_width_global

            last_statedot = statedot
            k = k + 1

        if len(obstacles) != 0:
            slack_obs = self.opti.variable(len(obstacles))
            self.__addSlackObsCost(slack_obs)
            self.opti.subject_to(slack_obs[:]>=0)
            for idx, obs in enumerate(obstacles):
                # obs = [center x, center y, height, width]
                dist_quad = (obs[2] ** 2 + obs[3] ** 2) ** 0.5 / 2.0 + self.robot_radius + self.safe_range_obs
                self.opti.subject_to((self.states[0,:]-obs[0])**4/dist_quad**4 + (self.states[1,:]-obs[1])**4/dist_quad**4 >= 1 - slack_obs[idx])

        finalState[2] = finalState[2] - self.safe_range_height
        self.opti.subject_to(self.states[[0,1,2,3,4,5,6,7], -1] - finalState[[0,1,2,3,4,5,6,7]] == self.slack_final[[0,1,2,3,4,5,6,7]])
        self.opti.subject_to(ca.cos(self.states[8, -1] - finalState[8]) == 1-self.slack_final[8])
        self.opti.subject_to(self.states[[9,10,11], -1] - finalState[[9,10,11]] == self.slack_final[[9,10,11]])
        self.__addSlackFinalCost(self.slack_final)
        self.__addSlackConvexCost(self.slack_convex)
        self.__addTimeCost()
        
        if (init_sol_dict):
            # Optimizing for style starting from base
            print("Adding Style Cost.")
            self.__addStyleCost(self.states, self.time)
            self.opti.set_initial(self.states, init_sol_dict['states'])
            self.opti.set_initial(self.inputs, init_sol_dict['inputs'])
        else:
            # Optimizing base
            X0 = self.__get_initial(initState, finalState)
            self.opti.set_initial(self.states, X0)

        self.opti.minimize(self.total_cost)

        try:
            sol = self.opti.solve()
            print("Successfully Solved")
        except:
            print("Resorted to Debug")
            sol = self.opti.debug
        print(f"Total Cost: {sol.value(self.total_cost):.1f}")
        if (init_sol_dict):
            print(f"Style Cost: {sol.value(self.style_cost):.1f}")

        sol_dict = dict()
        sol_dict['states'] = sol.value(self.states)
        sol_dict['inputs'] = sol.value(self.inputs)
        # NOTE: new variable
        sol_dict['total_time'] = sol.value(self.time)
        print(f"Final Time: {sol_dict['total_time']:.2f}")
        sol_dict['time'] = self.__get_time_axis(sol_dict['total_time'])
        return sol_dict

    def __get_initial(self, initState, finalState):
        total_time_guess = 0.05 * self.grid_num
        grid = np.linspace(0, 1, self.grid_num + 1)
        def get_grid_values(i, j):
            delta = finalState[i] - initState[i]
            v0 = initState[i] + grid * delta
            dv0 = np.hstack(
                [initState[j].reshape((1,)),
                np.squeeze(1.0 / total_time_guess * np.ones((1, self.grid_num - 1)) * delta),
                finalState[j].reshape((1,)),])
            return v0, dv0
        x0, dx0 = get_grid_values(0, 3)
        y0, dy0 = get_grid_values(1, 4)
        z0, dz0 = get_grid_values(2, 5)
        roll0, droll0 = get_grid_values(6, 9)
        pitch0, dpitch0 = get_grid_values(7, 10)
        yaw0, dyaw0 = get_grid_values(8, 11)
        X0 = np.vstack([x0, y0, z0, dx0, dy0, dz0, 
                        roll0, pitch0, yaw0, droll0, dpitch0, dyaw0])
        return X0


if __name__ == "__main__":
    slip_opti = SLIPOpti()

    SFP = '../../shared/'
    PARAMS_F = [SFP+"params.pkl"]
    ORDER_F = [SFP+'order.pkl']
    MOTION_F = [SFP+'motion.csv']
    TRAJ_F = [SFP+'traj.pkl']
    if TRANS == 'CLOUD':
        client = DevelopmentClient()
        raise Exception("Provide explicit box folder ids for all 3 below!")
        PARAMS_FOLDER = client.folder()
        ORDER_FOLDER = client.folder()
        RESULT_FOLDER = client.folder()

        PARAMS_F.append(PARAMS_FOLDER)
        ORDER_F.append(ORDER_FOLDER)
        MOTION_F.append(RESULT_FOLDER)
        TRAJ_F.append(RESULT_FOLDER)

    def download(filespec, delete=False):
        local_fp = filespec[0]
        tag = local_fp[local_fp.rfind('/')+1:local_fp.rfind('.')]
        print(f"Downloading {tag}")
        if TRANS == 'LFSYS':
            while not os.path.isfile(local_fp):
                print(f"Waiting For {tag}")
                time.sleep(POLL_FREQ)
        elif TRANS == 'CLOUD':
            box_folder = filespec[1]
            while True:
                try:
                    box_file_id = next(box_folder.get_items()).id
                    break
                except:
                    print(f"Waiting For {tag}")
                    time.sleep(POLL_FREQ)
            box_file = client.file(box_file_id)
            box_file.download_to(open(local_fp, "wb"))
            if delete:
                box_file.delete()
        
        data = pickle.load(open(local_fp, 'rb'))
        if delete:
            os.remove(local_fp)
        return data

    def upload(filespec, data=None):
        local_fp = filespec[0]
        tag = local_fp[local_fp.rfind('/')+1:local_fp.rfind('.')]
        print(f"Uploading {tag}")
        if data is not None:
            pickle.dump(data, open(local_fp, 'wb'))
        if TRANS == 'CLOUD':
            box_folder = filespec[1]
            box_folder.upload(local_fp)

    download(PARAMS_F)
    while True:
        order = download(ORDER_F, delete=True)
        if order == 'update_params':
            download(PARAMS_F)
            continue
        
        print("Constructing Environment")
        obs = order['task_dict']['cones']
        obs = np.concatenate((obs, 0.125 * np.ones([len(obs), 2])), axis = 1)
        initState = np.array(2*[0] + [0.98] + 9*[0])
        target_xy = list(order['task_dict']['target'])
        target_yaw = np.arctan2(target_xy[1], target_xy[0])
        finalState = np.array(target_xy + [0.98] + 5*[0] + [target_yaw] + 3*[0])

        print("Optimizing Base Trajectory")
        base_sd = slip_opti.solve(initState, finalState, obs)
        base_traj = slip_opti.get_style_traj(base_sd['states'], base_sd['total_time'])

        print("Optmizing Style Trajectory")
        style_target = order['target_style']
        slip_opti.style_cost_fn = get_cost_fn(style_target, N=base_traj[0].shape[1])
        style_sd = slip_opti.solve(initState, finalState, obs, base_sd)

        print("Uploading motion.csv")
        wmg = WalkingMotionGenerator()
        wmg.generate_wholebody_motion_from_opti(style_sd)
        wmg.export_motion_to_csv(MOTION_F[0])

        print("Uploading traj.pkl")
        def traj_dict(sol_dict):
            traj = slip_opti.get_style_traj(
                sol_dict['states'], sol_dict['total_time'])
            return {'states': traj[0].T, 'dTs': traj[1].T}
        traj = {'base': traj_dict(base_sd), 'style': traj_dict(style_sd)}
        
        upload(MOTION_F)
        upload(TRAJ_F, data=traj)
