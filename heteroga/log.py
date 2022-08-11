import re
import math
import os
import time
import glob
import json

import ase.io
import numpy as np

from geometry_check import log_similarity


class LogGenerator:
    def __init__(self,
                 conf=None,
                 remove_identical_mode='similarity',
                 log_msg=''):

        self.conf = conf
        self.mode = remove_identical_mode

        self.logfile = self.conf.logfile
        self.log_msg = log_msg

        self.atoms_galog_each_cluster = {}  # 每一个cluster每一代的结构
        self.energy_galog_each_cluster = {} # 每一个cluster每一代的能量
        self.rank_each_cluster = {}

        self.lowest_energy = self._set_lowest_energy()

    def create_log_each_gen(self, gen=0):

        self.atoms_galog_each_cluster = {}
        self.energy_galog_each_cluster = {}
        self.rank_each_cluster = {}

        for now_clu in range(0, self.conf.num_cluster):
            atoms_list, energy_list = self._galog_each_cluster(gen, now_clu, return_atom_list=True)
            self.atoms_galog_each_cluster[now_clu] = atoms_list
            self.energy_galog_each_cluster[now_clu] = energy_list

            self.rank_each_cluster[now_clu] = self._update_rank_each_cluster(gen, now_clu)

        self._generate_rank_file_each_gen(gen)
        self.save_log()

    def _galog_each_cluster(self, gen, now_clu, return_atom_list=False):
        '''
        Return atoms_list, energy_list
        The "input.arc" was collected.
        读取各个cluster下的GAlog.txt文件，并输出列表

        TODO: 或许后续还是用dict保存初始结构和末态结构比较好？
        :param gen:
        :param now_clu:
        :param return_atom_list:
        :return:
        '''
        path_clu = os.path.join(self.conf.home_path, 'Cluster' + str(now_clu))
        gen_path = os.path.join(path_clu, "Gen" + str(gen))
        file = open(os.path.join(gen_path, "GAlog.txt"), 'r')

        galog = [[int(x.split()[0]), int(x.split()[1]), float(x.split()[2])] for x in file]

        all_files = [os.path.join(gen_path, str(int(x[1])), "input.arc") for x in galog]

        if return_atom_list:
            atoms_list = [ase.io.read(x, format='dmol-arc') for x in all_files]
            energy_list = [a[2] for a in galog]
            return atoms_list, energy_list

        return galog

    def restart_log(self, gen):
        """
        For restart program and initialize the

        Args:
            gen:

        Returns:

        """
        if self.rank_each_cluster == {} or self.atoms_galog_each_cluster == {}:
            for now_clu in range(0, self.conf.num_cluster):
                atoms_list, energy_list = self._galog_each_cluster(gen - 1, now_clu, return_atom_list=True)
                self.atoms_galog_each_cluster[now_clu] = atoms_list
                self.energy_galog_each_cluster[now_clu] = energy_list
                self.rank_each_cluster[now_clu] = self._rank_each_cluster(gen - 1, now_clu)

    def genrate_basic_info(self, gen):
        # if os.path.exists(os.path.join(self.conf.home_path, self.logfile)):
        #     self.log_msg += '\n ———— End of Previous Task.———— \n'

        self.log_msg += '\n #### #### #### HeteroGA #### #### ####  \n\n'
        self.log_msg += f' - The symbol of optimized structure: {self.conf.stru[:self.conf.num_atom].symbols} \n'
        self.log_msg += f' - This task already finished optimizing {gen-1} generation. \n'
        self.log_msg += f' - Start Date: {time.asctime(time.localtime(time.time()))} \n\n'
        self.save_log()

    def _rank_each_cluster(self, gen, now_clu):
        '''
        读取某一代的rank文件并把rank_list输出出去
        :param gen:
        :param now_clu:
        :return:
        '''
        path_clu = os.path.join(self.conf.home_path, 'Cluster' + str(now_clu))
        rank_file = os.path.join(path_clu, f'rank_after_G{gen}.txt')

        if os.path.exists(os.path.join(path_clu, f'Gen{gen}')) and not os.path.exists(rank_file):
            self._update_rank_each_cluster(gen, now_clu)

        rank_fi = open(rank_file, "r")
        rank_file_list = [[int(x.split()[0]), int(x.split()[1]), float(x.split()[2])] for x in rank_fi]
        rank_fi.close()

        return rank_file_list

    def _update_rank_each_cluster(self, gen, now_clu):
        """
        The GAlog.txt contain the 0 structure that is transferred from other cluster.
        更新rank文件，把每一代算完的galog，查重后和上一代的rank合并，然后保存
        Args:
            gen:
            path_clu:
            return_galog_atom_list: bool
                If true, output the ase_atom and energies "generated" in this generation.
            return_rank:
                If ture, output the location information updated after each generation.
        Returns:

        """
        path_clu = os.path.join(self.conf.home_path, 'Cluster' + str(now_clu))
        ga_list = self._galog_each_cluster(gen, now_clu)

        if gen == 0:
            last_rank_list = list()
        else:
            last_rank_list = self._rank_each_cluster(gen - 1, now_clu)

        # remove_identical
        rank_list = self.penalty_identical(ga_list + last_rank_list, path_clu)
        rank_list = sorted(rank_list, key=(lambda x: x[2]))[:self.conf.num_fit]

        fit_file = open(os.path.join(path_clu, f'rank_after_G{gen}.txt'), "w")
        for n in range(len(rank_list)):
            fit_file.write("%4d%4d%15.7f\n" % (rank_list[n][0], rank_list[n][1], rank_list[n][2]))
        fit_file.close()

        return rank_list

    def _generate_rank_file_each_gen(self, gen):
        '''
        整理每个cluster下面的文件，然后生成每一代总的rank
        :param gen:
        :return:
        '''
        all_list = list()

        for i in range(0, self.conf.num_cluster):
            file = open(os.path.join(self.conf.home_path, f"Cluster{str(i)}", f'rank_after_G{str(gen)}.txt'))
            file_list = [[i, int(x.split()[0]), int(x.split()[1]), float(x.split()[2])] for x in file]
            file.close()
            all_list = all_list + file_list

        all_list = sorted(all_list, key=(lambda x: x[3]))
        self.log_msg += (f'Best Structure is at Cluster {all_list[0][0]} / Gen {all_list[0][1]} / Num {all_list[0][2]} '
                         f'with {all_list[0][3]} \n')

        self.save_best_stru(all_list[0])


    def penalty_identical(self, file_list, path_clu):
        file_list = delete_same_energy(file_list)

        if self.mode == 'similarity':  # TODO: support AHC etc method
            num = re.findall(r"\d+\.?\d*",path_clu)[-1]
            file_list, drop_list = log_similarity(file_list, path_clu, self.conf)
            self.log_msg += f' - Cluster {num} | ' \
                            f'Penalize {len(set(drop_list))} ({len(set(drop_list))/len(file_list) * 100:.2f} %) ' \
                            f'similar structure. \n'

        return file_list

    def save_log(self, initial=False):
        # https://www.codeleading.com/article/76532231942/ #
        with open(os.path.join(self.conf.home_path, self.logfile), "a") as log_file:
            if initial:
                log_file.truncate(0)
                log_file.write("Heteroga - Start Search. \n \n")

            log_file.write(self.log_msg)
            log_file.flush()
            self.log_msg = ''

    def save_best_stru(self, low_list):
        if float(low_list[3]) < self.lowest_energy:
            self.lowest_energy = float(low_list[3])

            best_stru = ase.io.read(os.path.join(self.conf.home_path,
                                                 f'Cluster{low_list[0]}/Gen{low_list[1]}/{low_list[2]}/CONTCAR'))
            save_site = os.path.join(self.conf.home_path, f'best_stru_{float(low_list[3]):.2f}.cif')
            ase.io.write(save_site, best_stru)
            self.log_msg += f'Save the lowest energy structure at {save_site}. \n'

    def _set_lowest_energy(self):
        for file in os.listdir(self.conf.home_path):
            if os.path.splitext(file)[1].lower() == '.cif':
                lowest_energy = -float(re.findall(r"\d+\.?\d*", file)[-1])
            else:
                lowest_energy = 0
        return lowest_energy

    def stru_generation_method_stat(self, conf, gen):
        stat = list()

        for now_cluster in range(conf.num_cluster):
            path_cluster = os.path.join(conf.home_path, 'Cluster' + str(now_cluster))
            info_json_list = glob.glob(os.path.join(path_cluster, 'Gen' + str(gen), '*', 'info.json'))

            for info_loc in info_json_list:
                with open(info_loc, 'r') as f:
                    info = json.load(fp=f)

                    try:
                        stat.append(info['Method'])
                    except KeyError:
                        self.log_msg += f'Error: cannot read {info_loc} \n'
                        pass

        dict_stat = {}
        for key in stat:
            dict_stat[key] = dict_stat.get(key, 0) + 1

        dict_stat = sorted(dict_stat.items(), key=lambda item: item[1], reverse=True)
        self.log_msg += f'{dict_stat} \n'

def delete_same_energy(raw_list):
    new_list = []
    energy_list = []
    for i in raw_list:
        if i[2] not in energy_list:
            energy_list.append(i[2])
            new_list.append(i)
    return new_list


def cal_fitness(list_output, num_fit, fit_alpha=-3, ahc_value=None):
    """

    Args:
        list_output:
        num_fit:
        fit_alpha:
        ahc_value:

    Returns:
        fitness_score: NumPy arrays
            A n×1 NumPy arrays, n = num_fit
    """
    if ahc_value is not None:
        num_fit = ahc_value['num_fit']

    de = list_output[num_fit - 1][2] - list_output[0][2]
    fitness_score = np.zeros(num_fit)

    for n in range(num_fit):
        fitness_score[n] = math.exp(fit_alpha * (list_output[n][2] - list_output[0][2]) / de)

    return fitness_score
