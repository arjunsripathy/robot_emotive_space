import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from nltk import pos_tag
from sklearn.cluster import KMeans
import pickle

from language_model import LanguageModel
from traj_cost_model import CostModel
from base_cache import BaseCache
from trajectory import StyleLatentLabel, StyleCostLabel

# StyleNetwork manages the discriminator and language model.

# Collection of models for optimizing robot style
class StyleNet():
    def __init__(self, env, traj_opt, retrain, config):
        '''
        env: an instance of the Env class for which we'd to optimize trajectories
        traj_opt: A TrajectoryOptimizer
        retrain: True/False flag whether we'd like to retrain our style model.
        config: StyleNetConfig instance
        '''
        self.env = env
        self.lang_model = LanguageModel(config)
        self.base_cache = traj_opt.base_cache

        self.min_style_propose = config.MIN_STYLE_PROPOSE
        self.curriculum_mode = config.CURRICULUM_MODE
        self.method = config.METHOD

        self.eval_words = config.EVAL_WORDS
        self.eval_reps = config.NUM_EVAL // len(config.EVAL_WORDS)
        self.fdbk_batch_size = config.FDBK_BATCH_SIZE
        self.sc_coef = config.SC_COEF
        self.reg_coef = config.REG_COEF
        self.style_latent_dim = config.STYLE_LATENT_DIM
        self.exp_seed = config.EXP_SEED
        np.random.seed(self.exp_seed)
        torch.manual_seed(self.exp_seed)
        
        self.style_words = [w for w in self.lang_model.vocab if self.is_style_word(w)]
        self.actionables = pickle.load(open(f"{config.SHARED_FP}/actionables.pkl", 'rb'))

        self.cost_model = CostModel(self.env, self.base_cache, config)
        if (not retrain):
            print("Loading previous models")
            self.cost_model.load_params()
        self.cost_model.save_params()

    def random_latent(self):
        # Return random full_latent_dim vector, values [-1, 1]
        latent = torch.rand(size=[self.full_latent_dim])
        return (2 * latent) - 1

    def language_style_latent(self, ann, noise_amt = 0.):
        '''
        ann: a natural language string representing some style

        Returns a tensor latent representation for the language model.
        '''
        return self.lang_model.style_latent(ann, noise_amt)

    def is_style_word(self, word):
        # Part of speech and threshold filter
        if pos_tag([word])[0][1] != 'JJ':
            return False
        se_norm = torch.norm(self.language_style_latent(word))
        return se_norm > self.min_style_propose

    def random_style_word(self):
        return self.style_words[np.random.randint(len(self.style_words))]

    def random_actionable(self):
        # Actionables are style words that are easy to visualize behaviors for
        return self.actionables[np.random.randint(len(self.actionables))]

    def get_style_curriculum(self, data_buffer=[], train=True, TOL=1e-6):
        '''
        data_buffer: human feedback so far, used  for active learning
        train: true if for training, false if for evaluation
        TOL: tolerance with generating optimal curriculum

        Samples emotions to base query trajectories on depending on curriculum mode.
        '''
        curriculum = []
        if not train:
            print("Eval Curriculum")
            rep_eval_words = self.eval_words * self.eval_reps
            for word in rep_eval_words:
                vad = self.language_style_latent(word).numpy()
                curriculum.append((vad, [word]))
            return curriculum

        MODE = self.curriculum_mode
        if len(data_buffer) == 0 and self.method == 'OURS':
            MODE = 'RANDOM_UNIFORM'

        K = self.fdbk_batch_size
        print(f"Train Curriculum size {K} {MODE}")
        vad_vals = []
        for word in self.style_words:
            vad_vals.append(self.language_style_latent(word))
        N = len(self.style_words)
        vad_vals = np.stack(vad_vals)

        def query(vad, num_reps=2):
            distance = lambda i: np.sum((vad_vals[i] - vad) ** 2)
            indices = sorted(range(N), key=distance)[:num_reps]
            return (vad, [f"{self.style_words[i]}" for i in indices])

        assert MODE in {'EVAL_SAMP', 'ACTIVE_STYLE', 
                        'DIVERSE_STYLE', 'RANDOM_UNIFORM'}
        if MODE == 'EVAL_SAMP':
            for word in np.random.choice(self.eval_words, size=K):
                vad = self.language_style_latent(word).numpy()
                curriculum.append((vad, [word]))
            return curriculum
        elif MODE == 'RANDOM_UNIFORM':
            for _ in range(K):
                vad = 2 * np.random.random([self.style_latent_dim]) - 1
                curriculum.append(query(vad))
            return curriculum

        
        kmeans = KMeans(n_clusters=K).fit(vad_vals)
        cluster_centers = kmeans.cluster_centers_

        if MODE == 'ACTIVE_STYLE':
            prev_styles = [dp.style_latent for dp in data_buffer 
                           if isinstance(dp, StyleLatentLabel)]
            if prev_styles:
                prev_styles = np.stack(prev_styles)
                opt_it = 0
                while True:
                    hyp_styles = np.concatenate((cluster_centers, prev_styles), axis=0)
                    sqd_dists = np.sum(vad_vals ** 2, axis=1, keepdims=True) - \
                                2 * vad_vals @ hyp_styles.T + \
                                np.sum(hyp_styles ** 2, axis=1, keepdims=True).T
                    nn_styles = np.argmin(sqd_dists, axis=1)
                    nn_one_hot = np.zeros([N, K + len(prev_styles)])
                    nn_one_hot[np.arange(N), nn_styles] = 1
                    adj_groups = nn_one_hot[:, :K]
                    adj_groups /= np.sum(adj_groups, axis=0, keepdims=True) + EPS
                    adj_cluster_centers = adj_groups.T @ vad_vals

                    err = np.mean((adj_cluster_centers - cluster_centers) ** 2)
                    if err < TOL:
                        print(f"Achieved {err:.1e}<{TOL:.1e} in {opt_it} additional iterations.")
                        break
                    else:
                        opt_it += 1
                        print(f"{opt_it} err={err:.1e}")
                        cluster_centers = adj_cluster_centers



        for i in range(K):
            curriculum.append(query(cluster_centers[i]))

        return curriculum


    def random_style_latent(self):
        return self.lang_model.style_latent(self.random_style_word())
    
    def get_style_cost_fn(self, target_style):
        # Returns style cost fn which may be optimized to achieve target style
        return self.cost_model.get_style_cost_fn(target_style, self.sc_coef)

    def predict_traj_style(self, traj):
        pred = self.cost_model.traj_model(traj).detach().numpy()
        if self.method == 'OURS':
            pred = np.mean(pred, axis=0)
        return pred

    def update_cost_model(self, fdbk_buffer, save_params, verbose=True):
        '''
        fdbk_buffer: same as data_buffer
        save_params: whether or not save parameters after updating

        Updates cost model parameters with human feedback.
        '''

        if verbose:
            print(f"Updating CM: {len(fdbk_buffer)} Data.", end="\r")

        fdbk_loss = 0
        if fdbk_buffer:
            fdbk_batch = []
            N = len(fdbk_buffer)

            for fdbk in fdbk_buffer:
                if self.method == 'OURS':
                    sc_fn = self.get_style_cost_fn(fdbk.style_latent)
                    cost = sc_fn(fdbk.traj) / self.sc_coef
                    fdbk_loss += cost
                elif self.method == 'ALLAN':
                    sc_fn = self.get_style_cost_fn(fdbk.style_lang)
                    cost = sc_fn(fdbk.traj) / self.sc_coef
                    fdbk_loss += (cost - fdbk.cost) ** 2
            fdbk_loss /= len(fdbk_buffer)

        reg_loss = self.cost_model.regularizer()

        self.cost_model.train(fdbk_loss + self.reg_coef * reg_loss)
        if save_params:
            self.cost_model.save_params()
        
        if fdbk_buffer:
            fdbk_loss = fdbk_loss.detach().numpy()
        reg_loss = reg_loss.detach().numpy()
        metrics = {'fdbk_loss': fdbk_loss, 'reg_loss': reg_loss}
    
        return metrics

    def in_vocab(self, ann):
        # see LanguageModel for more info
        return self.lang_model.in_vocab(ann)

    def format_style_latent(self, style_latent):
        # Displays Style Embdg in a readable format
        comps = []
        for i in range(self.style_latent_dim):
            feat = self.lang_model.features[i]
            val = style_latent[i]
            comps.append(f"{feat[:3]}: {val:.2f}")
        return '; '.join(comps)

EPS = 1e-6
# Class specifying all hyperparameters and configuration to use a StyleNet
class StyleNetConfig():
    # Potentially variable elements of config between experiments.
    # Other attributes should probably held constant.
    VARS = ['SC_COEF', 'CM_WAYPT_HID_DIMS', 'CM_TRAJ_HID_DIMS', 
            'CM_ENC_MODE', 'CM_DIST_METRIC', 'CM_ACTIVATION', 
            'CURRICULUM_MODE', 'FDBK_BATCH_SIZE', 'REG_COEF', 
            'EXT_OPT', 'LANG_PROJ_NOISE', 'METHOD',
            'NUM_EVAL', 'EVAL_PAIRS', 'ALLAN_LABEL_ALL']
    def __init__(self, STATE_DIM, LM):
        '''
        STATE_DIM: Number of dimensions in the state space
        LM: Lagrange multipliers to use

        Initializes default configurations, fields may be customized afterwards.
        Fields with no reasonable default value are required arguments.
        '''

        # Filesystem
        self.EXT_OPT = False # External trajectory optimizer?
        self.DATA_FP = None # Filepath to data will be set by interface
        self.FRAMES_FP = 'frames' # Filepath to save frames to for visuals
        self.SHARED_FP = '../../shared' # Folder shared across project

        # Basic
        self.STATE_DIM = STATE_DIM # Robot state dimension
        self.TIME_DIM = 1

        # Method
        self.METHOD = 'OURS' # Learning method {'OURS', 'ALLAN'}
        self.ALLAN_LABEL_ALL = False #ALLAN => SEP/SEP-ALL, toggles between the two.

        #Style Cost Model hyperparameters
        self.STYLE_LATENT_DIM = 3 # VAD dimension
        self.CM_WAYPT_HID_DIMS = [8] # Layer dimensions for waypt part of network
        self.CM_TRAJ_HID_DIMS = [16] # Layer dimensions for traj part of network
        self.CM_ENC_MODE = 'traj' # Determine we pool waypoints together or not
        self.CM_DIST_METRIC = 'SQD' # Distance metric to be optimized to hit target style
        self.CM_ACTIVATION = 'elu' # Network activation function

        # Language Model
        self.NO_STYLE = 'none' # Input you may provide to indicate 0 style
        self.LANG_PROJ_NOISE = 2e-3 # Noise we inject for variance to same language

        # Training configuration
        self.FDBK_BATCH_SIZE = 20 # B, number of labels per batch
        self.FDBK_ROUNDS = 2 #K, number of rounds of labeling
        self.TRAIN_LR = 1e-2 # Training learning rate
        self.REG_COEF = 2e-4 # L1 regularization coefficient
        self.MAX_TRAIN_ITER = 5000 # Maximum number of training updates with fixed data
        self.MAX_TRAIN_TIME = 5 * 60 # Maximum number of training seconds with fixed data

        # Trajectory optimization configuration
        self.SC_COEF = 1e5 # Alpha parameter, used to integrate style cost
        self.AVG_DT = 0.2 # Average time delta between waypoints
        '''
        Trajectory optimization first optimizes base traj (opt_base) then style traj
        opt_style. Below we specify the maximum number of iterations to optimize,
        how often to check if we've stagnated, and what cost reduction threshold we
        do better than to justify continuing.
        '''
        self.TO_MAX_ITERS = {'opt_base': 100, 'opt_style': 400}
        self.TO_CHECK_EVERY = {'opt_base': 20, 'opt_style': 80}
        self.TO_STOP_EARLY= {'opt_base': 0.95, 'opt_style': 0.975}
        self.TO_CKPT_EVERY = 5 # How often we checkpoint if best.
        self.SIG_LOWER = 0.995 # Best only if at least this much better.
        self.TO_MAX_SC_STOP = 1000 # Don't early stop unless style cost below
        LM = np.array(LM) # Lagrange multipliers
        self.LM = {'opt_base': LM, 'opt_style': LM}
        self.TO_LR = {'opt_base': 1e-2, 'opt_style': 1e-2} # Optimization learning rate
        self.LR_DECAY = 0.5 # Learning rate decay when we stagnate
        self.TIME_RANGE_FACTOR = 3 # Factor by which we allow deviation from AVG_DT

        # Training visualization parameters
        self.VIZ_FRAME_RATE = 1 # Frame rate for overlay visual
        self.VIZ_MAX_DECAY = 0.2 # Maximum alpha decay for earliest frame.

        # Curriculum parameters
        self.CURRICULUM_MODE = 'EVAL_SAMP' # Emotion sampling procedure
        self.MIN_STYLE_PROPOSE = 0.7 # Minimum threshold to be a style word
        self.EXP_SEED = 1 # Manual seed
        self.EVAL_PAIRS = [['sadness', 'joy'], 
                           ['fear', 'confidence'],
                           ['anger', 'patience']] #basic evaluation emotions
        self.NUM_EVAL = 18 # N * M, number of evaluation trajectories per phase

    @property
    def EVAL_WORDS(self):
        return sum(self.EVAL_PAIRS, [])

    def update_vars_from(self, filepath):
        # Update variable params stored in filepath
        saved_config = pickle.load(open(filepath, 'rb'))
        for var in StyleNetConfig.VARS:
            exec(f"self.{var} = saved_config.{var}")

    def export(self):
        # Export components required for external optimization
        return {'METHOD': self.METHOD,
                'STATE_DIM': self.STATE_DIM,
                'TIME_DIM': self.TIME_DIM,
                'STYLE_LATENT_DIM': self.STYLE_LATENT_DIM,
                'CM_ENC_MODE': self.CM_ENC_MODE,
                'CM_DIST_METRIC': self.CM_DIST_METRIC,
                'CM_ACTIVATION': self.CM_ACTIVATION,
                'CM_WAYPT_HID_DIMS': self.CM_WAYPT_HID_DIMS,
                'CM_TRAJ_HID_DIMS': self.CM_TRAJ_HID_DIMS}

    def print_vars(self):
        for var in StyleNetConfig.VARS:
            print(f"{var}: {getattr(self, var)}")


