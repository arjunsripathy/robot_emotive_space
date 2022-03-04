import casadi as ca
import pickle
import numpy as np

'''
CasADI implementation of the neural network style cost
function. Expects a params.pkl in SHARED_FP containing
the the relevant values of the Style Network configuration
as well as the actual parameters of the neural network itself.
'''

SHARED_FP = '../../shared'
EPS = 1e-8

S_NORM = np.array([ 50,  50, 1.0,   1,  1.,  0.5,
                    np.radians(5), np.radians(15),
                    np.radians(180), np.radians(5),
                    np.radians(15), np.radians(30)])

def get_cost_fn(style_target, N, coef=1e5):
    p = pickle.load(open(f'{SHARED_FP}/params.pkl', 'rb'))
    config = p['config']
    INP_DIM = config['STATE_DIM'] + config['TIME_DIM']

    W_HDs = [INP_DIM] + config['CM_WAYPT_HID_DIMS']
    T_INP_DIM = 2 * W_HDs[-1] if config['CM_ENC_MODE'] == 'traj' else W_HDs[-1]
    T_HDs = [T_INP_DIM] + config['CM_TRAJ_HID_DIMS']
    if config['METHOD'] == 'OURS':
        T_HDs.append(config['STYLE_LATENT_DIM'])

    def N_cast(vector):
        return ca.DM(np.expand_dims(vector, axis = 1).repeat(N, 1))

    def proc_traj(traj):
        states, dTs = traj # 12 x N, 1 x N
        yaw_cos, yaw_sin = ca.cos(states[8:9, :]), ca.sin(states[8:9, :])
        nstates = states / N_cast(S_NORM)
        return ca.vertcat(states[:8, :], yaw_cos, yaw_sin, states[9:, :], dTs) # 14 x N

    ACT_FNS = {'relu': lambda x: ca.fmax(x, 0.),
               'tanh': lambda x: ca.tanh(x),
               'leaky_relu': lambda x: ca.fmax(x, 0.) + 0.01 * ca.fmin(x, 0.),
               'elu': lambda x: ca.fmin(ca.fmax(x, 0.), ca.exp(x)-1),
               'identity': lambda x: x}
    act_fn = ACT_FNS[config['CM_ACTIVATION']]

    def layer(model_prefix, name, x, act=True):
        if model_prefix is None:
            weight, bias = p[f'{name}.weight'], p[f'{name}.bias']
        else:
            weight = p[model_prefix][f'{name}.weight']
            bias = p[model_prefix][f'{name}.bias']
        value = ca.DM(weight) @ x + N_cast(bias)
        return act_fn(value) if act else value

    def compute_latent(traj, model_prefix=None):
        x = proc_traj(traj)
        for i in range(len(W_HDs) - 1):
            x = layer(model_prefix, f'waypt_net.{2 * i}', x)
        if config['CM_ENC_MODE'] == 'traj':
            softmax_feats = ca.log(ca.sum2(ca.exp(x)))
            mean_feats = x @ np.ones([N, 1]) / N
            x = ca.vertcat(softmax_feats, mean_feats)
        for i in range(len(T_HDs) - 1):
            x = layer(model_prefix, f'out_net.{2 * i}', x, act = i < len(T_HDs) - 2)
        if config['METHOD'] == 'OURS':
            x = ca.tanh(x)
        return x

    def cost_fn(traj):
        S = config['STYLE_LATENT_DIM']
        if config['METHOD'] == 'OURS':
            traj_latent = compute_latent(traj)
            targ_latent = ca.repmat(style_target, (1, N))
            diff = traj_latent - targ_latent
            if config['CM_DIST_METRIC'] == 'SQD':
                dist = ca.sumsqr(diff) / (S * N)
            elif config['CM_DIST_METRIC'] == 'COS':
                S1s, N1s = np.ones([1, S]) , np.ones([N, 1])
                dist = (S1s @ (diff @ N1s))[0, 0]/ (S * N)
            return coef * dist
        elif config['METHOD'] == 'ALLAN':
            traj_latent = compute_latent(traj, model_prefix=style_target)
            return coef * ca.sumsqr(traj_latent) / (S * N)


    return cost_fn

if __name__ == '__main__':
    cost_fn = get_cost_fn(np.array([0, 0, 0]), 46)
    print(cost_fn((ca.DM.zeros(12, 46), ca.DM.zeros(1, 46))))




