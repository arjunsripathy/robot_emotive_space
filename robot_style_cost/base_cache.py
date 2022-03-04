import os
import pickle

# Caches base trajectory optimizations keyed by base_id
class BaseCache():
    def __init__(self, traj_opt, config):
        self.traj_opt = traj_opt
        self.filepath = f"{config.DATA_FP}/base_cache.pkl"
        try:
            self.cache = pickle.load(open(self.filepath, 'rb'))
        except:
            self.cache = dict()
        self.cache = dict()
        self.last_base = (None, None)

    def get_base_sol(self, traj, verbose=False):
        # Get's base solution for trajectory
        if traj.base:
            return (traj.base, None)
        base_id = traj.base_id
        if (base_id != self.last_base[0]):
            if (base_id not in self.cache):
                if (verbose):
                    print(f"Computing Base Traj {base_id[:40]}...")
                base_problem = self.traj_opt.get_base_problem(traj)
                sol = self.traj_opt.optimize_problem(base_problem, run_type = 'opt_base', verbose = verbose)
                self.cache[base_id] = sol
                pickle.dump(self.cache, open(self.filepath, 'wb'))
            self.last_base = (base_id, self.cache[base_id])

        base_traj = self.last_base[1]
        return base_traj

