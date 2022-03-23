import os
import copy
import math
import glob
import json
from abc import ABC, abstractmethod

import sklearn
import numpy as np
from scipy import stats

from gaussian_process import GPR


class SamplingMethod(ABC):
    def __init__(self, cluster_mode='Multi'):
        self.cluster_mode = cluster_mode

    @abstractmethod
    def initial(self):
        pass

    @abstractmethod
    def update(self, now_cluster):
        pass

    @abstractmethod
    def sample(self, candidate_list, number):
        pass


class GPRSampling(SamplingMethod):
    def __init__(self, log, conf, k=1):
        SamplingMethod.__init__(self)
        self.conf = conf

        self.gpr_model = {} # TODO: should support single model
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
                    self.update(now_cluster)

                self.gpr_model[now_cluster].train()

    def update(self, now_cluster):
        path_cluster = os.path.join(self.conf.home_path, 'Cluster' + str(now_cluster))
        atoms_list = self.log.atoms_galog_each_cluster[now_cluster]
        energy_list = self.log.energy_galog_each_cluster[now_cluster]
        self.gpr_model[now_cluster].memory.save_data(atoms_list, energy_list, save_loc=path_cluster)

    def sample(self, candidate_list, number, mode='thompson', now_cluster=None):
        '''

        Only the top layers structure will be sent to train.

        Args:
            candidate_list:
            number:
            mode:
            now_cluster:

        Returns:

        '''
        temp_list = list()
        if mode == 'lower_conf_bound':  # TODO: lower_conf_bound. not fixed yet
            cal_list = [i[:self.conf.num_atom] for i in candidate_list]
            e_up, e_std = self.gpr_model.predict_energy(cal_list, eval_std=True)
            acquisition_func = np.array(e_up) - self.k * np.array(e_std)
            append_list = [[e_up[i], acquisition_func[i], e_std[i]] for i in range(len(e_up))]

        elif mode == 'thompson':
            thom_list = [i[:self.conf.num_atom] for i in candidate_list]
            e_up, acquisition_func = self.gpr_model[now_cluster].thompson_sampling(thom_list)
            for i in range(len(thom_list)):
                temp_struture = copy.deepcopy(candidate_list[i])
                e_std = abs(acquisition_func[i] - e_up[i])

                temp_struture.info.update({'Predicted_Env': e_up[i]})
                temp_struture.info.update({'Predicted_Var': e_std})

                temp_list.append([temp_struture, e_up[i], acquisition_func[i]])

        temp_list = sorted(temp_list, key=(lambda x: x[2]), reverse=False)
        final_list = [item[0] for item in temp_list[:number]]
        return final_list

    def error_stat(self, conf, log, gen):
        for now_cluster in range(self.conf.num_cluster):
            path_cluster = os.path.join(conf.home_path, 'Cluster' + str(now_cluster))
            info_json_list = glob.glob(os.path.join(path_cluster, 'Gen' + str(gen), '*', 'info.json'))

            error_list = list()
            for info in info_json_list:
                with open(info, 'r') as f:
                    info = json.load(fp=f)
                    error_list.append([info['Predicted_Env'], info['Calculated_Env'], info['Predicted_Var']])

            pred_list = [x[0] for x in error_list]
            calc_list = [x[1] for x in error_list]

            rmse = math.sqrt(sklearn.metrics.mean_squared_error(pred_list, calc_list))
            tau, _ = stats.kendalltau(pred_list, calc_list)

            log.log_msg += f" - Cluster {now_cluster} | " \
                           f"Number of training structure: {len(self.gpr_model[now_cluster].memory.features):.2f} , " \
                           f"RMSE for this generation: {rmse:.2f} and Kentall tau: {tau:.2f}\n"
