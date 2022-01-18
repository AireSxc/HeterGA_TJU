import os
from functools import reduce
from abc import ABC, abstractmethod

import numpy as np

from gaussian_process import GPR, gpr_recode, load_gpr_recode

class SamplingMethod(ABC):
    def __init__(self, cluster_mode='Multi'):
        self.cluster_mode = cluster_mode

    @abstractmethod
    def initial(self):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def sample(self, candidate_list):
        pass

class GPRSampling(SamplingMethod):
    def __init__(self, log, conf, k=1):
        SamplingMethod.__init__(self)
        self.conf = conf

        self.gpr_model = {}
        self.log = log
        self.k = k

    def initial(self):
        if self.cluster_mode == 'Multi':  # TODO: for single cluster
            for now_cluster in range(self.conf.num_cluster):
                path_cluster = os.path.join(self.conf.home_path, 'Cluster' + str(now_cluster))
                self.gpr_model[now_cluster] = GPR(sample_stru=os.path.join(self.conf.home_path, self.conf.stru_file))

                if os.path.exists(os.path.join(path_cluster, 'GPR_para.npz')):
                    self.gpr_model[now_cluster].memory.load_para(os.path.join(path_cluster, 'GPR_para.npz'))
                else:
                    self.update(now_cluster, path_cluster)

                self.gpr_model[now_cluster].train()

    def update(self, now_cluster, path_cluster):
        atoms_list, energy_list = self.log.return_atom_energy_gen_list()
        self.gpr_model[now_cluster].memory.save_data(atoms_list, energy_list, save_loc=path_cluster)

    def sample(self, candidate_list, mode='thompson'):
        if mode == 'lower_conf_bound': # TODO: lower_conf_bound. not fixed yet
            cal_list = [i[0][:self.conf.num_atom] for i in candidate_list]
            e_up, e_std = self.model.predict_energy(cal_list, eval_std=True)
            acquisition_func = np.array(e_up) - self.k * np.array(e_std)
            append_list = [[e_up[i], acquisition_func[i], e_std[i]] for i in range(len(e_up))]

        elif mode == 'thompson':
            cal_list = [i[0][:self.conf.num_atom] for i in candidate_list]
            e_up, acquisition_func = self.model.thompson_sampling(cal_list)
            append_list = [[e_up[i], acquisition_func[i], 0] for i in range(len(e_up))]

        final_list = np.hstack((candidate_list, append_list))
        final_list = sorted(final_list, key=(lambda x: x[5]), reverse=False)

        return final_list



