import copy
import multiprocessing
import os
import random
from abc import ABC, abstractmethod

import ase.io
import numpy as np
from ase.data import atomic_numbers
from geometry_check import atoms_too_close, atoms_too_close_two_sets
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
            log.log_msg += f"Structural Calculation Raise Problem at Gen {g} Num {n} \n"
        return stru

    def parallel_frame(self, ope, path_gen, now_cluster, iter_list):
        # https://blog.csdn.net/qq_34914551/article/details/119451639 #
        pbar = tqdm(total=len(iter_list))
        pbar.set_description(f'Progress for Cluster {now_cluster}')
        update = lambda *args: pbar.update()

        parallel_run = multiprocessing.Pool(processes=multiprocessing.cpu_count())
        multiple_results = [parallel_run.apply_async(ope,
                                                     args=(item, path_gen),
                                                     callback=update)
                            for item in iter_list]

        [res.get() for res in multiple_results]

        parallel_run.close()
        parallel_run.join()


class Initializer(OffspringOperation):
    def __init__(self, conf, log_set):
        OffspringOperation.__init__(self)
        self.conf = conf
        self.box = box_generator(self.conf.stru_min, self.conf.top_num, self.blmin, 'Pt', self.conf.height_ratio)
        self.log = log_set

    def multi_cluster_initial(self, ope):
        for now_cluster in range(self.conf.num_cluster):
            self.log.log_msg += f" - Start Initialization Calculation at Cluster {now_cluster} \n"
            path_cluster = os.path.join(self.conf.home_path, 'Cluster' + str(now_cluster))
            path_gen = os.path.join(path_cluster, 'Gen0')

            iter_list = list(range(self.conf.initial_size))
            iter_list = restart_cofirm(path_gen, iter_list)

            os.chdir(path_gen)
            self.parallel_frame(ope, path_gen, now_cluster, iter_list)

    def run(self):
        self.log.save_log(initial=True)

        if self.conf.cluster_mode == 'Single':
            self.log.log_msg += 'Initialization with Single Cluster Mode \n'  # TODO
            # self.single_cluster()

        elif self.conf.cluster_mode == 'Multi':
            self.log.log_msg += 'Initialization with Multi-Cluster Mode \n'
            self.multi_cluster_initial(self.parallel_initializer)

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
        self.mp_list = multiprocessing.Manager().list()

        self.mutations = OperationSelector([1., 1.],
                                           [rattle(self.blmin, self.num_atom),
                                            permutation(self.blmin, self.num_atom)])

    def multi_cluster_following(self, op_selector):
        for now_cluster in range(self.conf.num_cluster):
            self.log.log_msg += f" - Start Offspring Calculation at Cluster {now_cluster} / Gen {self.gen}\n"
            path_cluster = os.path.join(self.conf.home_path, 'Cluster' + str(now_cluster))
            path_gen = os.path.join(path_cluster, 'Gen' + str(self.gen))
            os.chdir(path_gen)

            iter_list = list(range(self.conf.num_pop * self.conf.samp_ratio))
            self.parallel_frame(op_selector, path_gen, now_cluster, iter_list)

    def ga_op(self, now_cluster, limited_time=50):
        too_close = True
        mutate = ' '
        stru_temp = copy.deepcopy(self.conf.slab)
        stru_temp_pos = stru_temp.get_scaled_positions()

        list_for_cross = log_set.gen_loclist_with_cluster[now_cluster]
        path_cluster = os.path.join(self.conf.home_path, 'Cluster' + str(now_cluster))

        count = 0
        while too_close:
            count += 1
            if count > limited_time:
                break

            parent_array = return_parent_number(self.log, self.conf, now_cluster, num=2)
            p1 = parent_array[0]
            p2 = parent_array[1]

            parent = [p1, p2]
            parent.sort()

            if parent not in self.mp_list:
                pos_p1 = read_structure(list_for_cross[p1], path_cluster)
                pos_p2 = read_structure(list_for_cross[p2], path_cluster)
                new_stru_pos = crossover(pos_p1, pos_p2, stru_temp_pos, self.conf.num_elem_atom)

                if new_stru_pos is not None:
                    stru_temp.set_scaled_positions(new_stru_pos)
                    too_close = atoms_too_close(stru_temp[:self.conf.num_atom], self.conf.blmin)

                    if not too_close:
                        if random.randrange(0, 100) < self.conf.mutate_rate:
                            slab_new = None
                            count_time = 0

                            while slab_new is None and count_time < 20:
                                slab_new = mutations.get_structure(stru_temp)
                                count_time += 1

                        mp_list.append(parent)
                        candidate_list.append([slab_new, p1, p2, mutate])

    def parallel_following(self):

    def run(self):
        if self.conf.cluster_mode == 'Single':  # TODO
            self.log.log_msg += 'Initialization with Single Cluster Mode \n'

        elif self.conf.cluster_mode == 'Multi':
            for now_cluster in range(self.conf.num_cluster):
                exchange(self.gen, now_cluster, self.conf, self.log)
                op_selector = OperationSelector([0.8, 0.2],
                                                [ga_op(now_cluster), de_op(now_cluster)])
                self.multi_cluster_following(self.parallel_following, op_selector)

        self.log.create_log_each_gen()

        self.log.log_msg += f'Calculation at Gen {self.gen} Finished'
        self.log.save_log()

    # def parallel_offspring(self):


############## utilities #############################

class OperationSelector():
    """Class to produce new candidates by applying one of the
    candidate generation operations which is supplied in the
    'operations'-list. The operations are drawn randomly according
    to the 'probabilities'-list.

    operations : list or list of lists
        Defines the operations to generate new candidates in GOFEE.
        of mutations/crossovers. Either a list of mutations, e.g. the
        RattleMutation, or alternatively a list of lists of such mutations,
        in which case consecutive operations, one drawn from each list,
        are performed.

    probabilities : list or list of lists
        probability for each of the mutations/crossovers
        in operations. Must have the same dimensions as operations.
    # https://gitlab.au.dk/au480665/gofee #
    """

    def __init__(self, probabilities, operations):
        cond1 = isinstance(operations[0], list)
        cond2 = isinstance(probabilities[0], list)
        if not cond1 and not cond2:
            operations = [operations]
            probabilities = [probabilities]
        element_count_operations = [len(op_list) for op_list in operations]
        element_count_probabilities = [len(prob_list)
                                       for prob_list in probabilities]
        assert element_count_operations == element_count_probabilities, 'the two lists must have the same shape'
        self.operations = operations
        self.rho = [np.cumsum(prob_list) for prob_list in probabilities]

    def __get_index__(self, rho):
        """Draw from the cumulative probalility distribution, rho,
        to return the index of which operation to use"""
        v = np.random.random() * rho[-1]
        for i in range(len(rho)):
            if rho[i] > v:
                return i

    def get_new_candidate(self, parents):
        """Generate new candidate by applying a randomly drawn
        operation on the structures. This is done successively for
        each list of operations, if multiple are present.
        """
        for op_list, rho_list in zip(self.operations, self.rho):
            for i_trial in range(5):  # Do five trials
                to_use = self.__get_index__(rho_list)
                anew = op_list[to_use].get_new_candidate(parents)
                if anew is not None:
                    parents[0] = anew
                    break
            else:
                anew = parents[0]
                anew = op_list[to_use].finalize(anew, successfull=False)
        return anew


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


def restart_cofirm(path_gen, iter_list):
    if not os.path.exists(path_gen):
        os.makedirs(path_gen)
    else:
        previous_cal_list = list()

        if os.path.exists(os.path.join(path_gen, 'GAlog.txt')):
            file = open(os.path.join(path_gen, 'GAlog.txt'), 'r')
            previous_cal_list = [int(x.split()[1]) for x in file]

        for i in list(set(previous_cal_list)):
            iter_list.remove(i)

        self.log.log_msg += f" -  - Restart: Cluster {now_cluster} " \
                            f"Already Calculate {len(previous_cal_list)} Structures \n \n"
    return iter_list


def return_parent_number(log_set, conf, now_cluster, num=2):
    random_list = list(range(conf.num_fit))
    parent_array = np.zeros(num)
    list_for_comm = log_set.gen_loclist_with_cluster[now_cluster]
    list_fitness = cal_fitness(list_for_comm, conf.num_fit)

    for i in parent_array:
        while parent_array[i] == 0:
            p_t_temp = random.choice(random_list)
            if random.randrange(0, 10000) / 10000.0 < list_fitness[p_t]:
                parent_array[i] = p_t_temp
                random_list.remove(p_t_temp)

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
