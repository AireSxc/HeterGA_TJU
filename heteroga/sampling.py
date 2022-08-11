import os
import copy
import math
import glob
import json
from abc import ABC, abstractmethod

import numpy as np
from scipy import stats
import sklearn.metrics

from machine_learning import GPR


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

        self.log = log
        self.k = k
        self.gpr_model = GPR(self.conf, self.log)

    def initial(self):

        # if os.path.exists(os.path.join(self.conf.home_path, self.conf.data_save_file)):
        #     self.gpr_model.memory.load_dataset(os.path.join(self.conf.home_path, self.conf.data_save_file))
        #     if self.cluster_mode == 'Multi':  # TODO: for single cluster
        #         for now_cluster in range(self.conf.num_cluster):
        #                 self.update(now_cluster)
        # else:

        if os.path.exists(os.path.join(self.conf.home_path, self.conf.data_save_file)):
            self.gpr_model.memory.load_dataset(os.path.join(self.conf.home_path, self.conf.data_save_file))

        self.gpr_model.memory.dataset_nan_reload(self.conf)

        if os.path.exists(os.path.join(self.conf.home_path, self.conf.model_save_file)):
        #     self.gpr_model.train(request_train=False)
        # else:
            self.gpr_model.train()


    def update(self, now_cluster):
        # path_cluster = os.path.join(self.conf.home_path, 'Cluster' + str(now_cluster))
        atoms_list = self.log.atoms_galog_each_cluster[now_cluster]
        energy_list = self.log.energy_galog_each_cluster[now_cluster]
        self.gpr_model.memory.save_data(atoms_list, energy_list, self.conf.gen,
                                        save_loc=os.path.join(self.conf.home_path, self.conf.data_save_file))

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
            # e_up, e_std = self.gpr_model.predict_energy(cal_list, eval_std=True)
            acq_func = np.array(e_up) - self.k * np.array(e_std)
            append_list = [[e_up[i], acq_func[i], e_std[i]] for i in range(len(e_up))]

        elif mode == 'thompson':
            # thom_list = [i[:self.conf.num_atom] for i in candidate_list]
            e_up, std, acq_func = self.gpr_model.sampling(candidate_list)
            e_up = e_up.tolist()
            std = std.numpy().tolist()
            acq_func = acq_func.tolist()
            for i in range(len(candidate_list)):
                temp_struture = copy.deepcopy(candidate_list[i])
                temp_struture.info.update({'Predicted_Env': e_up[i]})
                temp_struture.info.update({'Predicted_Std': std[i]})
                temp_struture.info.update({'Acq_func': acq_func[i]})
                temp_list.append([temp_struture, e_up[i], acq_func[i]])

        temp_list = sorted(temp_list, key=(lambda x: x[2]), reverse=False)
        final_list = [item[0] for item in temp_list[:number]]
        return final_list

    def error_stat(self, conf, gen):
        for now_cluster in range(self.conf.num_cluster):
            path_cluster = os.path.join(conf.home_path, 'Cluster' + str(now_cluster))
            info_json_list = glob.glob(os.path.join(path_cluster, 'Gen' + str(gen), '*', 'info.json'))

            error_list = list()
            for info_loc in info_json_list:
                with open(info_loc, 'r') as f:
                    info = json.load(fp=f)

                    try:
                        error_list.append([info['Predicted_Env'], info['Calculated_Env'],
                                           info['Predicted_Std'], info['Acq_func']])
                    except KeyError:
                        self.log.log_msg += f'Error: cannot read {info_loc} \n'
                        pass

            self.log.save_log()

            pred_list = [x[0] for x in error_list]
            calc_list = [x[1] for x in error_list]

            rmse = math.sqrt(sklearn.metrics.mean_squared_error(pred_list, calc_list))
            tau, _ = stats.kendalltau(pred_list, calc_list)

            self.log.log_msg += f" - Cluster {now_cluster} | " \
                           f"RMSE for this generation: {rmse:.2f} and Kentall tau: {tau:.2f}\n"
