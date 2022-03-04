import pickle
import argparse
import numpy as np
import torch
import PIL
import matplotlib.pyplot as plt
import os
import time
from boxsdk import DevelopmentClient
from shutil import copyfile


from traj_opt import TrajectoryOptimizer
from style_net import StyleNet, StyleNetConfig
from trajectory import Trajectory, StyleLatentLabel, StyleCostLabel
import wandb

'''
Base Interface for style cost experiments, the extensions of these
are the main entry point for running experiments. Most of the experiment
code using the environment API to interact with the specifics of robots.
'''

class Interface():
    def __init__(self, env, config):
        '''
        env: instance of Env class, environment for experiments
        config: StyleNetConfig

        Initializes style network, trajectory optimizer, and  data buffer
        and everything else required for running experiments.
        '''
        parser = argparse.ArgumentParser(description='Robot Style Cost Interface')
        # Which experiment mode, defined in the run function below
        parser.add_argument('mode_index', help="Operation Mode Index")
        # Name used to reference data folder with parameters, labels, etc.
        # If 'allan' is in name then SEP/SEP-ALl is used, otherwise
        # our method is used.
        parser.add_argument('name', default="", help="model name")
        # Retrain neural network parameters
        parser.add_argument('--retrain', default=False, action='store_true')
        # Override previuosly saved config with currently passed in one.
        # Normally passed in config is only used if we are retraining
        parser.add_argument('--override_cfg', default=False, action='store_true')
        # Clear collected data buffer of human feedback labels
        parser.add_argument('--clear_buffer', default=False, action='store_true')
        # Manual random seed if desired
        parser.add_argument('--seed', type=int, default=None)
        # Use autolabeler/simulated human as opposed to querying for human feedback
        parser.add_argument('--auto', default=False, action='store_true')
        # N, number of evaluation emotions {2, 4, 6}
        parser.add_argument('--num_ems', type=int, default=None)
        # K * B, total number of human feedback to collect
        parser.add_argument('--num_fdbk', type=int, default=None)
        # N * M, total number of trajectories to evaluate on for each phase
        parser.add_argument('--num_eval', type=int, default=None)
        # Reuse evaluation trajectories between likert scale and choice phases
        parser.add_argument('--reuse_eval', default=False, action='store_true')
        # Silence some prints and confirmations
        parser.add_argument('--silent', default=False, action='store_true')
        # If using allan, this toggles between SEP & SEP-ALL
        parser.add_argument('--allan_label_all', default=False, action='store_true')
        # If using external traj opt (e.g. for Cassie), whether that's local or cloud
        parser.add_argument('--ext_opt_local', default=False, action='store_true')
        self.args = parser.parse_args()

        self.reuse_eval = self.args.reuse_eval
        self.silent = self.args.silent
        self.ext_opt_local = self.args.ext_opt_local
        self.poll_freq = 1 if self.ext_opt_local else 5
        self.initialized_ext_params = False

        self.data_fp = f"data_{self.args.name}"
        if not os.path.isdir(self.data_fp):
            if not self.silent:
                input(f"Make new dir: {self.data_fp}?")
            ind = 0
            while '/' in self.data_fp[ind:]:
                ind = self.data_fp.index('/')
                parent = self.data_fp[:ind]
                if not os.path.isdir(parent):
                    os.mkdir(parent)
                ind += 1
            os.mkdir(self.data_fp)

        self.config = config
        if 'allan' in self.data_fp:
            if 'orig_arc' in self.data_fp:
                print(f"Using ALLAN original architecture")
                self.config.CM_ENC_MODE = 'waypt'
                self.config.CM_WAYPT_HID_DIMS = [42, 21]
                self.config.CM_TRAJ_HID_DIMS = [21]
                self.config.CM_ACTIVATION = 'tanh'
            else:
                print("Using ALLAN our architecture")
            self.config.METHOD = 'ALLAN'
            self.config.ALLAN_LABEL_ALL = self.args.allan_label_all
            print(f"LABEL ALL: {self.config.ALLAN_LABEL_ALL}")
            self.config.CURRICULUM_MODE = 'EVAL_SAMP'
        else:
            print("Using Regular")

        if self.args.seed is not None:
            self.config.EXP_SEED = self.args.seed
            print(f"SEED {self.config.EXP_SEED}")
        if self.args.num_ems is not None:
            num_ems = self.args.num_ems
            assert num_ems in {2, 4, 6}
            self.config.EVAL_PAIRS = self.config.EVAL_PAIRS[:num_ems//2]
            print(f"{', '.join(self.config.EVAL_WORDS)}")
        if self.args.num_fdbk is not None:
            num_fdbk = self.args.num_fdbk
            assert num_fdbk % self.config.FDBK_BATCH_SIZE == 0
            self.config.FDBK_ROUNDS = num_fdbk // self.config.FDBK_BATCH_SIZE
            print(f"Feedback Rounds {self.config.FDBK_ROUNDS}")
        if self.args.num_eval is not None:
            num_eval = self.args.num_eval
            assert num_eval % len(self.config.EVAL_WORDS) == 0
            self.config.NUM_EVAL = num_eval
            print(f"# Eval: {self.config.NUM_EVAL}")

        self.config.DATA_FP = self.data_fp
        self.config_fp = self.data_fp + "/config.pkl"
        self.log_fp = self.data_fp + '/log.pkl'
        self.eval_trajs_fp = self.data_fp + '/eval_trajs.pkl'
        if self.args.retrain:
            if not self.silent:
                input("Reset Log?")
            pickle.dump([], open(self.log_fp, 'wb'))
            pickle.dump([], open(self.eval_trajs_fp, 'wb'))
        if self.args.retrain or self.args.override_cfg:
            if not self.silent:
                input("Reset Config?")
            pickle.dump(config, open(self.config_fp, 'wb'))
        self.config.update_vars_from(self.config_fp)
        self.log = pickle.load(open(self.log_fp, 'rb'))
        self.eval_trajs = pickle.load(open(self.eval_trajs_fp, 'rb'))

        self.env = env
        self.traj_opt = TrajectoryOptimizer(self.env, self.config, self.args.silent)
        self.style_net = StyleNet(self.env, self.traj_opt, self.args.retrain, self.config)
        
        self.frames_fp = self.config.FRAMES_FP
        self.env.frames_fp = self.frames_fp
        self.shared_fp = self.config.SHARED_FP
        self.method = self.config.METHOD
        self.allan_label_all = self.config.ALLAN_LABEL_ALL
        self.eval_pairs, self.eval_words = self.config.EVAL_PAIRS, self.config.EVAL_WORDS
        self.eval_reps = self.config.NUM_EVAL // len(self.config.EVAL_WORDS)

        self.EXT_OPT = self.config.EXT_OPT
        if self.EXT_OPT and not self.ext_opt_local:
            self.client = DevelopmentClient()
            raise Exception("Provide explicit box folder ids for all 3 below!")
            self.orders_folder = self.client.folder()
            self.results_folder = self.client.folder()
            self.params_folder = self.client.folder()

        if (self.args.clear_buffer):
            print("Clearing buffer")
            self.data_buffer = []
        else:
            print("Loading previous buffer")
            self.data_buffer = pickle.load(open(f'{self.data_fp}/data_buffer.pkl', "rb"))

    def save_feedback(self, traj, adj, fdbk):
        # Saves feedback fdbk for traj, generated for adjectives adj.
        if fdbk is None:
            return
        fdbk = torch.tensor(fdbk, dtype=torch.float32)
        if self.method == 'OURS':
            datapoint = StyleLatentLabel(traj, style_latent=fdbk)
        elif self.method == 'ALLAN':
            datapoint = StyleCostLabel(traj, style_lang=adj[0], cost=fdbk)
        self.data_buffer.append(datapoint)
        self.save_db()

    def generate_trajectory(self, task_dict=None, style_lang=None, style_latent=None,
                            verbose=False, seed=None, ret_style_cost_fn=False):
        '''
        task_dict: If provided att_dict for task we solve (else random)
        style_latent: If provided style latent we try to embody
        style_lang: If provided, style language we try to embody
        verbose: If true print target and visualize trajectory
        seed: If provided seed numpy/torch
        ret_style_cost_fn: Return the style cost function used

        NOTE: ideally only one of style_latent & lang should be provided (else we just
        use style_latent). If neither provided a random style is used.

        Returns a generated robot trajectory that alligns with the provided
        style_lang using the current models.
        '''
        if seed:
            seed = int(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
        task = self.env.get_task(task_dict)
        task_dict = task.att_dict

        if style_latent is None:
            if style_lang is None:
                style_lang = self.style_net.random_style_word()
            style_latent = self.style_net.language_style_latent(style_lang)
        if verbose:
            fmt_vad = self.style_net.format_style_latent(style_latent)
            print(f"Generating trajectory for {fmt_vad}")
        
        if self.method in {'OURS', 'ORACLE'}:
            target_style = style_latent
        elif self.method == 'ALLAN':
            target_style = style_lang
        style_cost_fn = self.style_net.get_style_cost_fn(target_style) 
        if self.EXT_OPT:
            if not self.initialized_ext_params:
                self.export_model()
                self.force_ext_param_update()
                self.initialized_ext_params = True

            if self.method == 'ORACLE':
                raise Exception("External Oracle not supported.")
            if self.method == 'OURS':
                target_style = target_style.numpy()
            order = {'task_dict': self.env.np_task_dict(task_dict), 
                     'target_style': target_style}
            pickle.dump(order, open(self.shared_fp+"/order.pkl", 'wb'))
            if self.ext_opt_local:
                while not (os.path.isfile(f"{self.shared_fp}/traj.pkl") and \
                        os.path.isfile(f"{self.shared_fp}/motion.csv")):
                    print("Waiting for Optimization Result.")
                    time.sleep(self.poll_freq)
            else:
                self.orders_folder.upload(f"{self.shared_fp}/order.pkl")
                print("Exported Optimization Order.")
                while True:
                    try:
                        result_ids = dict()
                        for file in self.results_folder.get_items():
                            result_ids[file.name] = file.id
                        assert len(result_ids) == 2 # motion.csv, traj.pkl
                        break
                    except:
                        print("Waiting for Optimization Result.")
                        time.sleep(self.poll_freq)
                for file_name in result_ids:
                    file = self.client.file(result_ids[file_name])
                    file.download_to(open(self.shared_fp+"/"+file_name, 'wb'))
                    file.delete()

            opt = pickle.load(open(f"{self.shared_fp}/traj.pkl", 'rb'))
            for kind in opt:
                for k in opt[kind]:
                    opt[kind][k] = torch.tensor(opt[kind][k], dtype=torch.float32)
            traj = Trajectory(task, None, opt['style']['states'], 
                              opt['style']['dTs'], seed)
            traj.base = Trajectory(task, None, opt['base']['states'], 
                                   opt['base']['dTs'], seed)
            traj.csv_string = open(f"{self.shared_fp}/motion.csv", 'r').read()

            os.remove(f"{self.shared_fp}/traj.pkl")
            os.remove(f"{self.shared_fp}/motion.csv")

        else:       
            traj = self.traj_opt.plan(task, style_cost_fn)

        if verbose:
            print("AUTOLABEL")
            print(self.style_net.format_style_latent(self.env.autolabel(traj)))
            #self.visualize_motion(traj)

        if ret_style_cost_fn:
            return traj, style_cost_fn
        else:
            return traj

    def get_label(self):
        # Query human for feedback depending on the method

        if self.method == 'OURS':
            label = np.zeros(3)
            for i, query in enumerate(['Valence', 'Arousal', 'Dominance']):
                while True:
                    try:
                        value = float(input(f"{query}? "))
                        assert abs(value) <= 1
                    except:
                        continue
                    label[i] = value
                    break

        elif self.method == 'ALLAN':
            label = None
            while True:
                try:
                    label = float(input(f"How {self.config.ALLAN_STYLE_LANG}?"))
                except:
                    continue

        confirm = None
        fmt_label = self.style_net.format_style_latent(label)
        while confirm not in {'y', 'n'}:
            confirm = input(f"Confirm Label: {fmt_label}: ")

        if confirm == 'y':
            return label
        else:
            return self.get_label()


    def visualize_motion(self, traj, save_frames=None):
        '''
        Visualizes a trajectory, saving frames if desired. Frame
        saving is only supported for local optimization. For external
        optimization Blender will be used for visualization.
        '''

        if self.EXT_OPT:
            viz_folder = f"{self.shared_fp}/blender_viz"
            motion_csv = open(f"{viz_folder}/motion.csv", "w")
            motion_csv.write(traj.csv_string)
            motion_csv.close()
            task_dict = self.env.np_task_dict(traj.task_dict)
            pickle.dump(task_dict, open(f'{viz_folder}/task_dict.pkl', 'wb'))
            print(f"Use blender!")
            while True:
                if input("Proceed? ") == 'y':
                    break
        else:
            start_x = traj.init_state[0]
            goal_x = traj.init_state[-1]
            print(f"{start_x:.0f}->{goal_x:.0f} ({traj.time:.1f}s)")
            while True:
                self.env.visualize_trajectory(traj, save_frames=save_frames)
                if input("Proceed? ") == 'y':
                    break

    def collect_feedback(self, train=True, auto=False):
        # Generates & saves a series of annotated robot trajectories
        print(f"COLLECT FEEDBACK TRAIN={train}")
        if not train:
            torch.manual_seed(0)
            np.random.seed(0)

        curriculum = self.style_net.get_style_curriculum(self.data_buffer, train=train)
        queries, hopes, N = [], [], len(curriculum)

        record = ["Train" if train else "Eval", [], [], []]
        for style_query in curriculum:
            vad, adj = style_query
            fmt_vad = self.style_net.format_style_latent(vad)
            descr = f"{fmt_vad} ({', '.join(adj)})"
            if self.method in {'OURS', 'ORACLE'}:
                vad_t = torch.tensor(vad, dtype=torch.float32)
                traj = self.generate_trajectory(style_latent=vad_t)
            elif self.method == 'ALLAN':
                traj = self.generate_trajectory(style_lang=adj[0])
            queries.append((descr, vad, adj, traj))
            record[1].append(vad)
            print(f"Generated {len(queries)}/{N}")

        for i, query in enumerate(queries):
            print(f"Query {i+1}/{N}")
            descr, vad, adj, traj = query
            if not auto:
                input(descr)
            if auto:
                label = self.env.autolabel(traj, fmt='np')
            else:
                self.visualize_motion(traj)
                label = self.get_label()
            record[2].append(label)

            if self.method == 'OURS':
                feedback = label
            elif self.method == 'ALLAN':
                feedback = np.mean((label - vad) ** 2)

            if self.method == 'OURS':
                record[3].append(self.style_net.predict_traj_style(traj))
            datapoint = (traj, adj, feedback)
            if train:
                self.save_feedback(*datapoint)
                if self.method == 'ALLAN' and self.allan_label_all:
                    for other_adj in self.eval_words:
                        if other_adj == adj[0]:
                            continue
                        other_vad = self.style_net.language_style_latent(other_adj)
                        other_fdbk = np.mean((label - other_vad.numpy()) ** 2)
                        other_datapoint = (traj, [other_adj], other_fdbk)
                        self.save_feedback(*other_datapoint)


        self.log_record(record)


    def save_db(self):
        print(f"Saving New Data Buffer (Size: {len(self.data_buffer)})")
        pickle.dump(self.data_buffer, open(f'{self.data_fp}/data_buffer.pkl', "wb"))

    def log_record(self, record):
        self.log.append(record)
        pickle.dump(self.log, open(self.log_fp, 'wb'))

    def export_model(self):
        '''
        Exports model parameters as well as StyleNetConfig values relevant
        for external optimization.
        '''
        params = dict()

        if self.method == 'OURS':
            traj_msd = self.style_net.cost_model.traj_model.state_dict()
            for k in traj_msd:
                params[k] = traj_msd[k].detach().numpy()
        elif self.method == 'ALLAN':
            traj_models = self.style_net.cost_model.traj_models
            for model_k in traj_models:
                state_dict = traj_models[model_k].state_dict()
                params[model_k] = dict()
                for param_k in state_dict:
                    params[model_k][param_k] = state_dict[param_k].detach().numpy()

        params['config'] = self.config.export()

        pickle.dump(params, open(f'{self.shared_fp}/params.pkl', 'wb'))

        if not self.ext_opt_local:
            try:
                file_id = next(self.params_folder.get_items()).id
                file = self.client.file(file_id)
                file.delete()
            except:
                print("NOTE: No previous parameters to delete.")

            self.params_folder.upload(f"{self.shared_fp}/params.pkl")

    def create_visual(self):
        '''
        Creates a trajectory visualization by saving frames. For external
        optimization simply visualizes and expects you to save frames in
        Blender directly.
        '''
        style_lang = input("Prompt: ")
        traj = self.generate_trajectory(style_lang=style_lang, verbose=True)
        if self.EXT_OPT:
            self.visualize_motion(traj)
        else:
            frame_rate = self.config.VIZ_FRAME_RATE
            self.visualize_motion(traj, save_frames=frame_rate)
            self.combine_frames()

    def combine_frames(self):
        '''
        Overlay saved frames to produce a single image capturing how a
        trajectory evolves over time.
        '''
        imgs = []
        img_fps = [fp for fp in os.listdir(self.frames_fp) if '.png' in fp]
        img_fps = sorted(img_fps, key=lambda x: int(x[:x.index('.')]))
        for i, img_fp in enumerate(img_fps):
            if ('.png' not in img_fp):
                continue
            pil_img = PIL.Image.open(f"{self.frames_fp}/{img_fp}")
            imgs.append(np.array(pil_img, dtype = np.float32)/255.)
        imgs = np.stack(imgs)
        deviance = np.max(imgs, axis = 0) - imgs
        deviance = np.sum(deviance, axis=-1, keepdims=True)
        last_img_bias = np.reshape([0] * (len(imgs) - 1) + [1], [-1, 1, 1, 1])
        sig_dev = (deviance > 1e-1).astype(dtype=np.float32)
        sig_dev_share = sig_dev/(np.sum(sig_dev, axis=0, keepdims=True) + 1e-5)
        any_sig_dev = np.max(sig_dev, axis=0, keepdims=True)
        signal_mask = any_sig_dev * sig_dev_share + (1 - any_sig_dev) * last_img_bias
        MAX_DECAY = self.config.VIZ_MAX_DECAY
        N = len(imgs)
        decay = np.reshape([MAX_DECAY ** ((len(imgs) - 1 - i)/(N-1)) for i in range(len(imgs))], [-1, 1, 1, 1])
        imgs = ((1 - decay)  + decay * imgs) * signal_mask
        overlay = np.sum(imgs, axis = 0)
        title = input("Title? ")
        plt.imsave(f'visuals/overlay_{title}.png', overlay)

    def force_ext_param_update(self):
        # Forces external optimizer to update model parameters.
        self.export_model()
        order = "update_params"
        pickle.dump(order, open(f"{self.shared_fp}/order.pkl", 'wb'))
        if self.ext_opt_local:
            while os.path.isfile(f"{self.shared_fp}/order.pkl"):
                print("Waiting Update Params Order to be taken.")
                time.sleep(self.poll_freq)
        else:
            self.orders_folder.upload(f"{self.shared_fp}/order.pkl")
            print("Exported Update Params Order.")
            while True:
                try:
                    order = next(self.orders_folder.get_items()).id
                    print("Waiting Update Params Order to be taken.")
                    time.sleep(self.poll_freq)
                except:
                    break


    def train(self, auto=False):
        # Train model with collected human feedback
        print("TRAIN")

        raise Exception("If you'd like to use wandb provide values!")
        wandb.init(project=project, entity=entity, resume=False)
        start_time, its = time.time(), 0

        while (True):
            its += 1
            if its > self.config.MAX_TRAIN_ITER or \
               time.time() - start_time > self.config.MAX_TRAIN_TIME:
                break

            save_params = (its % 100 == 0)
            metrics = self.style_net.update_cost_model(self.data_buffer, save_params)

            wandb.log(metrics)

        if self.EXT_OPT:
            self.force_ext_param_update()

    def run_model(self):
        '''
        Run model generating a set of trajectories for a style query.
        '''
        NUM_PER_TASK = 1
        NUM_TRAJ = NUM_PER_TASK * 1
        while True:
            target_style_lang = input("Prompt? ")
            if target_style_lang == '':
                while True:
                    target_style_lang = self.style_net.random_actionable()
                    vad = self.style_net.language_style_latent(target_style_lang)
                    fmt_vad = self.style_net.format_style_latent(vad)
                    if input(f"{target_style_lang} ({fmt_vad})? ") == 'y':
                        break
            trajs = []
            task_dict = None
            for i in range(NUM_TRAJ):
                print(f"Traj {i + 1}/{NUM_TRAJ}")
                traj = self.generate_trajectory(task_dict=task_dict,
                        style_lang=target_style_lang, verbose=True)
                task_dict = traj.task_dict if i % NUM_PER_TASK == 0 else None
                trajs.append(traj)

    def analyze_log(self, verbose=True):
        # Analyze log from an experiment run.
        if verbose:
            self.config.print_vars()
            print("-" * 30)

        scale_export, top1_export, top2_export = [], [], []
        metric_fns = {'MSE': lambda x, y: np.mean((x-y) ** 2),
                      'MAE': lambda x, y: np.mean(np.abs(x-y)),
                      'BIN': lambda x, y: np.mean(x*y) < 0}
        train_labels_list = []
        for i, record in enumerate(self.log):
            print(f"Round {i+1} ({record[0]})")
            if record[0] in {'Train', 'Eval'}:
                targets, labels, preds = record[1:]
                N = len(targets)
                print(f"{N} labeled datapoints")
                if N == 0:
                    continue
                for name, fn in metric_fns.items():
                    val = np.mean([fn(targets[i], labels[i]) for i in range(N)])
                    s = f"{name} Label<>Targ: {val:.2f}"
                    if preds:
                        pred_targ = np.mean([fn(targets[i], preds[i]) for i in range(N)])
                        s+= f"; Pred<>Targ: {pred_targ:.2f})"
                        pred_val = np.mean([fn(labels[i], preds[i]) for i in range(N)])
                        s+= f"; Pred<>Label: {pred_val:.2f})"
                    print(s)

                if self.method == 'ALLAN':
                    continue

                if record[0] == 'Train':
                    train_labels_list.append(labels)
                else:
                    continue
                tla = np.concatenate(train_labels_list, axis=0)
                _, svs, _ = np.linalg.svd(tla - np.mean(tla, axis=0, keepdims=True))
                stds = np.sqrt(svs/len(tla))
                fmt_stds = ', '.join([f'{std:.2f}' for std in stds])
                print(f"Singular Values: [{fmt_stds}]")

                msds = []
                N = 10000
                for i in range(N):
                    latent = self.style_net.random_style_latent()
                    latent = np.expand_dims(latent.numpy(), axis=0)
                    diffs = latent - tla
                    min_dist = np.sqrt(np.min(np.sum(diffs ** 2, axis=1)))
                    msds.append(min_dist)
                print(f"Label AMD {np.mean(msds):.2f} +/- {2*np.std(msds)/np.sqrt(N):.2f}")
            elif record[0] == "Scale Results":
                results = record[1]
                print("Averages")
                for i in range(len(results)):
                    conf_rad = 2 * np.std(results[i]) / np.sqrt(len(results[i]))
                    print(f"Result {i+1} {np.mean(results[i]):.2f} +/- {conf_rad:.2f}")
                    scale_export.append(np.mean(results[i]))
                print("All")
                for i in range(len(results)):
                    val_list = ' '.join([f"{v:.1f}" for v in results[i]])
                    print(f"Result {i+1} All:  " + val_list)
            elif record[0] == "Choice Results":
                results = record[1]
                print("Accuracy")
                for i in range(len(results)):
                    correct1, correct2, count = 0, 0, 0
                    for e in results[i]:
                        correct1 += sum([r == e for r in results[i][e][0]])
                        correct2 += sum([r == e for r in results[i][e][1]])
                        count += len(results[i][e][0])
                    print(f"Result {i+1} Top1 Acc: {correct1}/{count}")
                    print(f"Result {i+1} Top2 Acc: {correct1+correct2}/{count}")
                    top1_export.append(correct1/count)
                    top2_export.append((correct1+correct2)/count)
                print("First Choices")
                for i in range(len(results)):
                    print(f"Result {i+1} All:")
                    for e in results[i]:
                        print(f"\t{e}: {', '.join(results[i][e][0])}")

        return scale_export, top1_export, top2_export

    def save_generations(self):
        '''
        Save trajectory generations with current model for human
        evaluation later on.
        '''
        print("Saving Trajectories to Evaluate")
        N, K = len(self.eval_words), 2 * self.eval_reps
        eval_traj_batch = dict()
        for i in range(N * K):
            eval_word = self.eval_words[i // K]
            print(f"Saving Traj #{i+1}/{N * K} ({eval_word})")
            if self.reuse_eval and i % K >= (K//2):
                traj = eval_traj_batch[f"{eval_word}_{(i % K)-(K//2)}"]
            else:
                traj = self.generate_trajectory(style_lang=eval_word)
            eval_traj_batch[f"{eval_word}_{i % K}"] = traj

        self.eval_trajs.append(eval_traj_batch)
        pickle.dump(self.eval_trajs, open(self.eval_trajs_fp, 'wb'))

        record = ["Eval", [], [], []]
        self.log_record(record)

    def evaluate_generations(self, auto=False):
        '''
        Evaluate trajectories saved using save_generations with likert
        scale question and choice based responses.
        '''
        print("Evaluating Trajectories to Evaluate")
        N, K, W = len(self.eval_trajs), self.eval_reps, len(self.eval_words)
        scale_results = np.zeros([N, K * W])

        eval_groups = dict()
        for e_low, e_high in self.eval_pairs:
            group = sum([sum([[(i, e_low, j), (i, e_high, j)] 
                           for j in range(K)], [])
                             for i in range(N)], [])
            eval_groups[(e_low, e_high)] = group

        scale_eval_order = []
        for e_low, e_high in np.random.permutation(self.eval_pairs):
            group = eval_groups[(e_low, e_high)]
            print(f"Rate 1 {e_low}<->{e_high} 7")
            if not auto:
                input("Ready to Scale?")
            for group_ind in np.random.permutation(2 * N * K):
                i, e, j = group[group_ind]
                scale_eval_order.append((i, e, j))
                if not self.silent:
                    print(f"Rating #{len(scale_eval_order)}/{N * K * W}")
                traj = self.eval_trajs[i][f"{e}_{j}"]
                
                if auto:
                    label = self.env.autolabel(traj, fmt='np')
                    l_low = self.style_net.language_style_latent(e_low).numpy()
                    l_high = self.style_net.language_style_latent(e_high).numpy()
                    diff = l_high - l_low
                    score = np.sum(diff * (label - l_low)) / np.sum(diff * diff)
                    rating = max(1, min(1  + 6 * score, 7))
                else:
                    self.visualize_motion(traj)
                    while True:
                        try:
                            rating = int(input("Rating: "))
                            assert rating >= 1 and rating <= 7
                        except:
                            continue
                        break
                if e == e_low:
                    rating = 8 - rating
                scale_results[i][K * self.eval_words.index(e) + j] = rating

        choice_results = [{e: [[], []] for e in self.eval_words} for _ in range(N)]
        eval_keys = sum([sum([[(i, e, j + K) for j in range(K)]
                        for i in range(N)], [])
                            for e in self.eval_words], [])

        if not auto:
            input("Ready to Choose?")
        choice_eval_order = []
        for eval_ind in np.random.permutation(N * K * W):
            i, e, j = eval_keys[eval_ind]
            choice_eval_order.append((i, e, j))
            traj = self.eval_trajs[i][f"{e}_{j}"]

            print(f"Choosing #{len(choice_eval_order)}/{N * K * W}")
            if auto:
                label = self.env.autolabel(traj, fmt='np')
                def dist(cand):
                    vad = self.style_net.language_style_latent(cand).numpy()
                    return np.linalg.norm(label - vad)
                sorted_emotions = sorted(self.eval_words, key=dist)
                first_choice, second_choice = sorted_emotions[:2]
            else:
                self.visualize_motion(traj)
                while True:
                    print(f"Pick two of [{', '.join(self.eval_words)}]")
                    first_choice = input("1: ")
                    if first_choice not in self.eval_words:
                        continue
                    second_choice = input("2: ")
                    if second_choice not in self.eval_words or \
                        second_choice == first_choice:
                        continue
                    break
            choice_results[i][e][0].append(first_choice)
            choice_results[i][e][1].append(second_choice)

        self.log_record(["Scale Results", scale_results, scale_eval_order])
        self.log_record(["Choice Results", choice_results, choice_eval_order])


    def run(self):
        '''
        Main entry point for interacting with the stye cost environment for purposes
        of teaching, training, or visualzing the current model.
        '''
        MODE_NAMES = ['Trajectory Label', #0
                      'Train', #1
                      'Run Model', #2
                      'Analyze Log', #3
                      'Run Experiment', #4
                      'Conduct Study', #5
                      'Export Model', #6
                      'Create Visual', #7
                      'Combine Frames'] #8

        MODE = MODE_NAMES[int(self.args.mode_index)]
        print(f"Executing {MODE}")

        auto = self.args.auto
        if (MODE == 'Trajectory Label'): # Collect one fedback batch
            self.collect_feedback(auto=auto)
        elif (MODE == 'Train'): # Train model with collected feedback
            self.train()
        elif (MODE == 'Run Model'): # Generate trajectories for style
            self.run_model()
        elif (MODE == 'Analyze Log'): # Analyze experiment log
            self.analyze_log()
        elif (MODE == 'Run Experiment'): # Run full experiment
            '''
            Used for conducting simulated human trials.
            Saves initial generations and after each round of training.
            Evaluates all at the end.
            '''
            for i in range(self.config.FDBK_ROUNDS):
                self.save_generations()
                self.collect_feedback(auto=auto)
                self.train(auto=auto)
            self.save_generations()
            self.evaluate_generations(auto=auto)
            self.analyze_log(verbose=False)
        elif (MODE == 'Conduct Study'): # Run full user study
            '''
            Used for conducting real human studies
            Only evaluates on trajectories generated after all label
            collection is complete.
            '''
            for i in range(self.config.FDBK_ROUNDS):
                self.collect_feedback(auto=auto)
                self.train(auto=auto)
            self.save_generations()
            self.evaluate_generations(auto=auto)
            self.analyze_log(verbose=False)
        elif (MODE == 'Export Model'): # Export Model
            self.export_model()
        elif (MODE == 'Create Visual'): # Save frames for visual
            self.create_visual()
        elif (MODE == 'Combine Frames'): # Overlay saved frames
            self.combine_frames()
        else:
            raise Exception('No command executed')

