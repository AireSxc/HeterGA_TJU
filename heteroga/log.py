import copy
import math
import os

import ase.io
import numpy as np
from geometry_check import cum_diff, looks_like


class LogGenerator:
    def __init__(self,
                 conf=None,
                 remove_identical_mode='similarity',
                 logfile='search.log',
                 log_msg=''):

        self.conf = conf
        self.mode = remove_identical_mode

        self.logfile = logfile
        self.log_msg = log_msg

        self.atoms_gen_list_each_cluster = {}
        self.energy_gen_list_each_cluster = {}
        self.gen_loclist_with_cluster = {}

    def create_log_each_gen(self, gen=0):
        self.atoms_gen_list_each_cluster = {}
        self.energy_gen_list_each_cluster = {}
        self.gen_loclist_with_cluster = {}

        for now_clu in range(0, self.conf.num_cluster):
            path_clu = os.path.join(self.conf.home_path, 'Cluster' + str(now_clu))
            atoms_list, energy_list = self.each_cluster_logfile(gen, path_clu, return_atom_energy_list=True)
            self.atoms_gen_list_each_cluster[now_clu] = atoms_list
            self.energy_gen_list_each_cluster[now_clu] = energy_list

            self.gen_loclist_with_cluster[now_clu] = self.each_cluster_logfile(gen, path_clu, return_file_list=True)

        self.all_logfile(gen)
        self.save_log()

    def detect_null_tuple_and_create_log(self, gen):
        if self.gen_loclist_with_cluster == {}:
            for now_clu in range(self.conf.num_cluster):
                path_clu = os.path.join(self.conf.home_path, 'Cluster' + str(now_clu))
                self.gen_loclist_with_cluster[now_clu] = self.each_cluster_logfile(gen - 1, path_clu,
                                                                                   return_file_list=True)

    def return_atom_energy_gen_list(self):
        return self.atoms_gen_list_each_cluster, self.energy_gen_list_each_cluster

    def all_logfile(self, gen):
        all_list = list()

        for i in range(0, self.conf.num_cluster):
            file = open(self.conf.home_path + "/Cluster" + str(i) + '/fitafterG' + str(gen) + ".txt")
            file_list = [[i, int(x.split()[0]), int(x.split()[1]), float(x.split()[2])] for x in file]
            file.close()
            all_list = all_list + file_list

        all_list = sorted(all_list, key=(lambda x: x[3]))
        self.log_msg += (f'Best Structure is Cluster {all_list[0][0]} / Gen {all_list[0][1]} / Num {all_list[0][2]} '
                         f'with {all_list[0][3]} \n')

    def each_cluster_logfile(self, gen, path_clu, return_atom_energy_list=False, return_file_list=False):
        """
        The GAlog.txt contain the 0 structure that is transferred from other cluster.

        Args:
            gen:
            path_clu:
            return_atom_energy_list: bool
                If true, output the ase_atom and energies generated in this generation.
            return_file_list:
                If ture, output the location information updated after each generation.
        Returns:

        """
        gen_path = os.path.join(path_clu, "Gen" + str(gen))
        file = open(os.path.join(gen_path, "GAlog.txt"), 'r')

        file_list = [[int(x.split()[0]), int(x.split()[1]), float(x.split()[2])] for x in file]
        file_list = sorted(file_list, key=(lambda x: x[2]))[:self.conf.num_fit]
        file_list_galog = copy.deepcopy(file_list)

        if gen > 0:
            fit_file_old = open(os.path.join(path_clu, f'fitafterG{gen - 1}.txt'), "r")
            fit_file_list = [[int(x.split()[0]), int(x.split()[1]), float(x.split()[2])] for x in fit_file_old]
            file_list = delete_energy_repeat(file_list + fit_file_list)

            # remove_identical
            file_list = remove_identical(home_path, file_list, path_clu)
            file_list = sorted(file_list, key=(lambda x: x[2]))[:self.conf.num_fit]

            fit_file_old.close()

        fit_file = open(os.path.join(path_clu, f'fitafterG{gen}.txt'), "w")

        for n in range(len(file_list)):
            fit_file.write("%4d%4d%15.7f\n" % (file_list[n][0], file_list[n][1], file_list[n][2]))

        file.close()
        fit_file.close()

        if return_atom_energy_list:
            all_files = [os.path.join(gen_path, str(int(x[1])) + "/input.arc") for x in file_list_galog]
            atoms_list = [ase.io.read(x, format='dmol-arc') for x in all_files]
            energy_list = [a[2] for a in file_list_galog]
            return atoms_list, energy_list

        if return_file_list:
            return file_list

    def remove_identical(self, file_list, path_clu):
        if self.mode == 'similarity':
            file_list = self.similarity(file_list, path_clu)

        return file_list

    def similarity(self, file_list, path_clu, delta_de=0.1, pair_cor_cum_diff=0.03):
        drop_list = list()
        file_list = sorted(file_list, key=(lambda x: x[2]), reverse=True)
        for i in range(len(file_list)):
            # TODO, read different structure type.
            path1 = path_clu + "/Gen" + str(int(file_list[i][0])) + "/" + str(int(file_list[i][1])) + "/CONTCAR"
            a1_stru = ase.io.read(path1, format='vasp')
            a1 = [a1_stru, file_list[i][2]]

            for j in range(i):
                path2 = path_clu + "/Gen" + str(int(file_list[j][0])) + "/" + str(int(file_list[j][1])) + "/CONTCAR"
                a2_stru = ase.io.read(path2, format='vasp')[:num_atom]
                a2 = [a2_stru, file_list[j][2]]
                similarity = looks_like(a1, a2, num_atom, delta_de, pair_cor_cum_diff)

                if similarity:
                    drop_list.append(j)

        for i in set(drop_list):
            file_list[i][2] = file_list[i][2] + 10

        self.log_msg += f'Drop {len(set(drop_list))} similar structure. \n'

        return file_list

    def save_log(self, initial=False):
        # https://www.codeleading.com/article/76532231942/ #
        with open(os.path.join(self.conf.home_path, self.logfile), "a") as log_file:
            if initial:
                log_file.truncate(0)
                log_file.write("Heteroga - Start Search. \n \n")

            print(self.log_msg)
            log_file.write(self.log_msg)
            log_file.flush()
            self.log_msg = ''


############## utilities #############################

def delete_energy_repeat(raw_list):
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
