import copy
import multiprocessing
import os
import random
from abc import ABC

import ase.io
import numpy as np
from ase.data import atomic_numbers
from geometry_check import atoms_too_close, atoms_too_close_two_sets
from log import cal_fitness
from multitribe import exchange
from tqdm import tqdm
from utilities import rearrange_order


class OffspringOperation(ABC):
    def __init__(self):
        self.gen = 0

    def _calculation_with_error_except(self, stru, log):
        stru.calc = self.conf.calc
        if isinstance(self.conf.calc, LaspCalculator):
            try:
                stru.get_potential_energy()
            except ValueError:
                os.rename("all.arc", "best.arc")
                stru.calc.read_energy()
                stru.calc.update_atoms()
            log.log_msg += f"Structural Calculation Raise Problem at Gen {self.gen} Num {self.gen} \n"
        return stru

    def parallel_frame(self, ope, iter_list, mode='calc', path_gen=None, now_cluster=None):
        # https://blog.csdn.net/qq_34914551/article/details/119451639 #
        pbar = tqdm(total=len(iter_list))
        pbar.set_description(f'Progress')
        update = lambda *args: pbar.update()

        parallel_run = multiprocessing.Pool(processes=multiprocessing.cpu_count())
        if mode == 'calc':
            multiple_results = [parallel_run.apply_async(ope,
                                                         args=(item, path_gen),
                                                         callback=update)
                                for item in iter_list]
        elif mode == 'sampling':
            multiple_results = [parallel_run.apply_async(ope(now_cluster), callback=update) for _ in iter_list]

        [res.get() for res in multiple_results]

        parallel_run.close()
        parallel_run.join()


class Initializer(OffspringOperation):
    def __init__(self, conf, log_set):
        OffspringOperation.__init__(self)
        self.conf = conf
        self.box = box_generator(self.conf.stru_min, self.conf.top_num, self.blmin, 'Pt', self.conf.height_ratio)
        self.log = log_set

    def initial_mode(self, ope, now_cluster):
        self.log.log_msg += f" - Start Initialization Calculation at Cluster {now_cluster} \n"
        path_cluster = os.path.join(self.conf.home_path, 'Cluster' + str(now_cluster))
        path_gen = os.path.join(path_cluster, 'Gen0')

        iter_list = list(range(self.conf.initial_size))
        iter_list = restart_cofirm(path_gen, iter_list, self.log)

        os.chdir(path_gen)
        self.parallel_frame(ope, iter_list, path_gen=path_gen)

    def run(self):
        self.log.save_log(initial=True)

        if self.conf.cluster_mode == 'Single':
            self.log.log_msg += 'Initialization with Single Cluster Mode \n'
            self.initial_mode(self.parallel_initializer, 0)

        elif self.conf.cluster_mode == 'Multi':
            self.log.log_msg += 'Initialization with Multi-Cluster Mode \n'
            for now_cluster in range(self.conf.num_cluster):
                self.initial_mode(self.parallel_initializer, now_cluster)

        self.log.create_log_each_gen(num_fit=self.conf.num_fit,
                                     num_cluster=self.conf.num_cluster)

        self.log.log_msg += 'Initialization Finished'
        self.log.save_log()

    def parallel_initializer(self, item, path_gen):
        stru_min_child = copy.deepcopy(self.conf.stru_min)

        subpath = os.path.join(path_gen, str(item))
        if not os.path.exists(str(subpath)):
            os.mkdir(str(subpath))

        os.chdir(subpath)

        if self.conf.initial_mode == 'total_random':
            new_stru_min = self._total_random_operator(stru_min_child, self.box)
        # TODO other initial_mode

        new_stru_min.calc = self.conf.calc
        try:
            new_stru_min.get_potential_energy()
        except ValueError:
            self.parallel_initializer(item, path_gen)

        ase.io.write("CONTCAR", new_stru_min, format='vasp')
        ase.io.write("output_0_" + str(item) + '.cif', new_stru_min)
        os.system("echo 0 " + str(item) + " " + str(new_stru_min.calc.results['energy']) + " >> ../GAlog.txt")

    def _total_random_operator(self, stru, box):
        slab_child_pos = stru.get_scaled_positions()

        stru_not_good = True
        while stru_not_good:
            for i in range(self.conf.num_atom_min):
                slab_child_pos[i][0] = random.randrange(0, 10000) / 10000.0
                slab_child_pos[i][1] = random.randrange(0, 10000) / 10000.0
                slab_child_pos[i][2] = random.randrange(0, 10000) / 10000.0 * box[0] + box[1]
                stru.set_scaled_positions(slab_child_pos)

            top = rearrange_order(stru[:self.conf.num_atom_min] * self.conf.supercell_num)
            down = stru[self.conf.num_atom_min:] * self.conf.supercell_num

            if not atoms_too_close(top, self.blmin, self.conf.adsorp_atom):
                stru_not_good = atoms_too_close_two_sets(top, down, self.blmin)

        slab_temp = top + down
        return slab_temp


class FollowingGen(OffspringOperation):
    def __init__(self, gen, conf, log_set):
        OffspringOperation.__init__(self)
        self.gen = gen

        self.log = log_set
        self.log.detect_null_tuple_and_create_log(self.gen)

        self.conf = conf
        self.conf.follow_initial()

        self.candidate_list = multiprocessing.Manager().list()

        self.parent_ga_list = multiprocessing.Manager().list()
        self.parent_de_list = multiprocessing.Manager().list()

        # self.mutations = OperationSelector([1., 1.],
        #                                    [rattle(self.blmin, self.num_atom),
        #                                     permutation(self.blmin, self.num_atom)])

        self.op_selector = OperationSelector([0.8, 0.2], [self.ga_op, self.de_op])

    # def multi_cluster_following(self, ope):
    #     for now_cluster in range(self.conf.num_cluster):
    #         self.log.log_msg += f" - Start Offspring Calculation at Cluster {now_cluster} / Gen {self.gen}\n"
    #         path_cluster = os.path.join(self.conf.home_path, 'Cluster' + str(now_cluster))
    #         path_gen = os.path.join(path_cluster, 'Gen' + str(self.gen))
    #         os.chdir(path_gen)
    #
    #         iter_list = list(range(self.conf.num_pop * self.conf.samp_ratio))
    #         self.parallel_frame(ope, path_gen, now_cluster, iter_list)

    def ga_op(self, now_cluster):
        method = 'ga'
        stru_temp = copy.deepcopy(self.conf.slab)
        stru_temp_pos = stru_temp.get_scaled_positions()

        list_for_cross = self.log.gen_loclist_with_cluster[now_cluster]
        path_cluster = os.path.join(self.conf.home_path, 'Cluster' + str(now_cluster))

        too_close = True
        count = 0
        while too_close:
            count += 1
            if count > 50:
                break

            parent_array = return_parent_number(self.log, self.conf, now_cluster, num=2)
            p1 = parent_array[0]
            p2 = parent_array[1]

            parent = [p1, p2]
            parent.sort()

            if parent not in self.parent_ga_list:
                pos_p1 = read_structure(list_for_cross[p1], path_cluster)
                pos_p2 = read_structure(list_for_cross[p2], path_cluster)
                new_stru_pos = crossover(pos_p1, pos_p2, stru_temp_pos, self.conf.num_elem_atom)

                if new_stru_pos is not None:
                    stru_temp.set_scaled_positions(new_stru_pos)
                    too_close = atoms_too_close(stru_temp[:self.conf.num_atom], self.conf.blmin)

                    if not too_close:
                        # if random.randrange(0, 100) < self.conf.mutate_rate:
                        #     stru_temp = self.mutations.get_structure(stru_temp)
                        #     method = 'ga_with_mutate'

                        stru_temp.label = {'Parent': parent, 'Method': method}
                        return stru_temp

    def de_op(self, now_cluster):
        too_close = True
        count = 0

        while too_close:
            count += 1
            if count > 50:
                break

            de_method = random.choice([de_rand_1, de_best_1, de_rand_to_best_1])
            stru_temp, parent, method = de_method(self.log, self.conf, now_cluster, self.parent_de_list)

            if stru_temp is not None:
                if not atoms_too_close_two_sets(stru_temp[:self.conf.num_atom], stru_temp[self.conf.num_atom:],
                                                self.conf.blmin):
                    if not atoms_too_close(stru_temp, self.conf.blmin):
                        stru_temp.label = {'Parent': parent, 'Method': method}
                        return stru_temp

    def random_ope_sampling(self, now_cluster):
        accept = False
        while accept is False:
            op, index = self.op_selector.get_operator()
            new_stru = op(now_cluster)
            if new_stru is not None:
                if 'ga' in new_stru.label['Method']:
                    self.parent_ga_list.append(new_stru.label['Parent'])
                elif 'de' in new_stru.label['Method']:
                    self.parent_de_list.append(new_stru.label['Parent'])

                accept = True
                print(new_stru.label['Method'])
                print(new_stru.label['Parent'])
                self.candidate_list.append(new_stru)

    def run(self):
        if self.conf.cluster_mode == 'Single':  # TODO
            self.log.log_msg += 'Initialization with Single Cluster Mode \n'

        elif self.conf.cluster_mode == 'Multi':
            for now_cluster in range(self.conf.num_cluster):
                iter_list = range(self.conf.num_pop * self.conf.samp_ratio)
                print(iter_list)
                self.parallel_frame(self.random_ope_sampling, iter_list, now_cluster=now_cluster, mode='sampling')

                exchange(self.gen, now_cluster, self.conf, self.log)

        self.log.create_log_each_gen()

        self.log.log_msg += f'Calculation at Gen {self.gen} Finished'
        self.log.save_log()


############## utilities #############################

class OperationSelector(object):
    """Class used to randomly select a procreation operation
    from a list of operations.

    Parameters:

    probabilities: A list of probabilities with which the different
        mutations should be selected. The norm of this list
        does not need to be 1.

    oplist: The list of operations to select from.
    """

    def __init__(self, probabilities, oplist):
        assert len(probabilities) == len(oplist)
        self.oplist = oplist
        self.rho = np.cumsum(probabilities)

    def __get_index__(self):
        v = random.random() * self.rho[-1]
        for i in range(len(self.rho)):
            if self.rho[i] > v:
                return i

    def get_operator(self):
        """Choose operator and return it."""
        to_use = self.__get_index__()
        return self.oplist[to_use], to_use


def box_generator(atoms, top_num, blmin, label, height_ratio):
    # cell = atoms.get_cell()
    # slab_min_pos = atoms.get_positions()
    #
    # zmax = max(slab_min_pos[top_num:atoms.get_global_number_of_atoms(), 2])
    # height_gap = list(set(atoms.get_positions()[:, 2]))[-1] - list(set(atoms.get_positions()[:, 2]))[-2]
    # min_z = zmax + height_gap / cell[2][2]  # assume z axis perpendicular to xy plane
    # d_z = height_gap * 1.8 / cell[2][2]
    # if d_z > 0.9:
    #     d_z = 0.9
    # box = [d_z, min_z]

    cell = atoms.get_cell()
    slab_min_pos = atoms.get_positions()
    zmax = max(slab_min_pos[top_num:atoms.get_global_number_of_atoms(), 2])
    height_gap = cal_height_gap(atoms, label, blmin)
    min_z = (zmax + 1) / cell[2][2]  # assume z axis perpendicular to xy plane
    d_z = height_gap * height_ratio / cell[2][2]  # One layer

    if min_z + d_z > 0.9:
        print('too high')

    box = [d_z, min_z]
    return box


def cal_height_gap(atoms, label, blmin):
    tmp = [atom.position[2] for atom in atoms if atom.symbol == label]
    for i in range(len(tmp)):
        if tmp[0] - tmp[i] > blmin[(atomic_numbers[label], atomic_numbers[label])]:
            return tmp[0] - tmp[i]


def restart_cofirm(path_gen, iter_list, log):
    if not os.path.exists(path_gen):
        os.makedirs(path_gen)
    else:
        previous_cal_list = list()

        if os.path.exists(os.path.join(path_gen, 'GAlog.txt')):
            file = open(os.path.join(path_gen, 'GAlog.txt'), 'r')
            previous_cal_list = [int(x.split()[1]) for x in file]

        for i in list(set(previous_cal_list)):
            iter_list.remove(i)

        log.log_msg += f" -  - Restart: Cluster {now_cluster} " \
                       f"Already Calculate {len(previous_cal_list)} Structures \n \n"
    return iter_list


def return_parent_number(log_set, conf, now_cluster, num=2,
                         return_stru=False, return_best_stru=False):
    path_cluster = os.path.join(conf.home_path, 'Cluster' + str(now_cluster))

    random_list = list(range(conf.num_fit))
    parent_array = np.zeros(num, dtype=np.int64)
    list_for_comm = log_set.gen_loclist_with_cluster[now_cluster]
    list_fitness = cal_fitness(list_for_comm, conf.num_fit)

    for i in range(len(parent_array)):
        accept = False
        while accept is False:
            p_t_temp = random.choice(random_list)
            if random.randrange(0, 10000) / 10000.0 < list_fitness[p_t_temp]:
                parent_array[i] = int(p_t_temp)
                random_list.remove(p_t_temp)
                accept = True

    if return_stru is True:
        return_stru_list = list()
        if return_best_stru is True:
            parent_array[0] = 0
        for item in parent_array:
            path = os.path.join(path_cluster, "Gen" + str(int(list_for_comm[item][0])),
                                str(int(list_for_comm[item][1])), "CONTCAR")
            slab = ase.io.read(path, format='vasp')
            return_stru_list.append([item, slab])
        return return_stru_list
    else:
        return parent_array


def read_structure(stru_info, home_path):
    stru_path = os.path.join(home_path, "Gen" + str(int(stru_info[0])), str(int(stru_info[1])), "CONTCAR")
    stru = ase.io.read(stru_path, format='vasp')
    return stru.get_scaled_positions()


def crossover(pos_p1, pos_p2, slab_pos, num_elem_atom, maxcount=100):
    num_elem = len(num_elem_atom)
    accept = False
    count = 0
    while not accept and count < maxcount:
        count += 1
        # Generate the line which devide the surface into two pieces
        p1 = [random.randrange(10000, 90000) / 100000.0, random.randrange(10000, 90000) / 100000.0]
        p2 = [0.5, 0.5]
        # p2 = copy.deepcopy(p1)
        # while math.fabs(p2[0] - p1[0]) < 0.001: # the line should not perpendicular to x axis
        #     p2 = [random.randrange(10000, 90000)/100000.0, random.randrange(10000, 90000)/100000.0]

        if p2[0] - p1[0] != 0:
            k = (p2[1] - p1[1]) / (p2[0] - p1[0])
            b = 0.5 - k * 0.5
            count_id = 0
            n_elem = 0
            while n_elem < num_elem:
                pos = []
                n_atom = count_id

                for n in range(num_elem_atom[n_elem]):
                    if pos_p1[n_atom][1] > k * pos_p1[n_atom][0] + b:
                        pos.append(pos_p1[n_atom])
                    if pos_p2[n_atom][1] < k * pos_p2[n_atom][0] + b:
                        pos.append(pos_p2[n_atom])
                    n_atom = n_atom + 1

                if len(pos) < num_elem_atom[n_elem]:
                    n_elem = num_elem
                else:
                    n_atom = count_id

                    for n in range(num_elem_atom[n_elem]):
                        slab_pos[n_atom] = copy.deepcopy(pos[n])
                        n_atom = n_atom + 1

                    count_id = count_id + num_elem_atom[n_elem]
                    n_elem = n_elem + 1

                    if n_elem == num_elem:
                        accept = True
    if count == 100:
        return None

    return slab_pos


def de_rand_1(log_set, conf, now_cluster, parent_de_list):
    return_stru_list = return_parent_number(log_set, conf, now_cluster, num=3, return_stru=True)
    num_1, stru_1 = return_stru_list[0][0], return_stru_list[0][1]
    top_1 = stru_1[:conf.num_atom]
    num_2, stru_2 = return_stru_list[1][0], return_stru_list[1][1]
    top_2 = stru_2[:conf.num_atom]
    num_3, stru_3 = return_stru_list[2][0], return_stru_list[2][1]
    top_3 = stru_3[:conf.num_atom]

    parent = sorted([num_1, num_2, num_3])
    if parent not in parent_de_list:
        x_diff = top_1.get_positions() - top_2.get_positions()
        v_donor = top_3.get_positions() + (random.randint(1, 50) / 100) * x_diff
        top_3.set_positions(v_donor)
        final = top_3 + stru_3[conf.num_atom:]
    return final, parent, 'de_rand_1'


def de_best_1(log_set, conf, now_cluster, parent_de_list):
    return_stru_list = return_parent_number(log_set, conf, now_cluster, num=3,
                                            return_stru=True, return_best_stru=True)
    num_best, stru_best = return_stru_list[0][0], return_stru_list[0][1]
    top_best = stru_best[:conf.num_atom]
    num_1, stru_1 = return_stru_list[1][0], return_stru_list[1][1]
    top_1 = stru_1[:conf.num_atom]
    num_2, stru_2 = return_stru_list[2][0], return_stru_list[2][1]
    top_2 = stru_2[:conf.num_atom]

    parent = sorted([0, num_1, num_2])

    if parent not in parent_de_list:
        x_diff = top_1.get_positions() - top_2.get_positions()
        v_donor = top_best.get_positions() + (random.randint(1, 50) / 100) * x_diff
        top_best.set_positions(v_donor)
        final = top_best + stru_best[conf.num_atom:]
    return final, parent, 'de_best_1'


def de_rand_to_best_1(log_set, conf, now_cluster, parent_de_list):
    return_stru_list = return_parent_number(log_set, conf, now_cluster, num=4,
                                            return_stru=True, return_best_stru=True)

    num_best, stru_best = return_stru_list[0][0], return_stru_list[0][1]
    top_best = stru_best[:conf.num_atom]
    num_1, stru_1 = return_stru_list[1][0], return_stru_list[1][1]
    top_1 = stru_1[:conf.num_atom]
    num_2, stru_2 = return_stru_list[2][0], return_stru_list[2][1]
    top_2 = stru_2[:conf.num_atom]
    num_3, stru_3 = return_stru_list[3][0], return_stru_list[3][1]
    top_3 = stru_3[:conf.num_atom]

    parent = sorted([0, num_1, num_2, num_3])

    if parent not in parent_de_list:
        x_diff_1 = top_best.get_positions() - top_3.get_positions()
        x_diff_2 = top_1.get_positions() - top_2.get_positions()
        v_donor = top_3.get_positions() + (random.randint(1, 20) / 100) * x_diff_1 + (
            random.randint(1, 20) / 100) * x_diff_2
        top_best.set_positions(v_donor)
        final = top_best + stru_best[conf.num_atom:]
    return final, parent, 'de_rand_to_best_1'
