import copy
import json
import pickle
import multiprocess
#https://stackoverflow.com/questions/8804830/python-multiprocessing-picklingerror-cant-pickle-type-function/21345423#21345423

import os
import random
import shutil
from abc import ABC

import ase.io
import numpy as np
from tqdm import tqdm
from ase.data import atomic_numbers
import ase.build
from ase.ga.cutandsplicepairing import CutAndSplicePairing
from ase.ga.utilities import (atoms_too_close, atoms_too_close_two_sets,
                              gather_atoms_by_tag)

from log import cal_fitness
from multitribe import exchange
from mutation import rattle, permutation, mirror
from utilities import rearrange_order


class OffspringOperation(ABC):
    def __init__(self):
        self.gen = 0

    def parallel_frame(self, ope, iter_list, task='calc', path_gen=None, now_cluster=None):
        # https://blog.csdn.net/qq_34914551/article/details/119451639 #
        pbar = tqdm(total=len(iter_list))
        update = lambda *args: pbar.update()
        pro_num = int(multiprocess.cpu_count()/9)
        parallel_run = multiprocess.Pool(processes=pro_num)

        if task == 'calc':
            pbar.set_description(f'Progress for calc')
            multiple_results = [parallel_run.apply_async(ope,
                                                         args=(item, path_gen, now_cluster),
                                                         callback=update)
                                for item in iter_list]
        elif task == 'sampling':
            pbar.set_description(f'Progress for sampling')
            multiple_results = [parallel_run.apply_async(ope,
                                                         args=(now_cluster,),
                                                         callback=update)
                                for _ in iter_list]

        [res.get() for res in multiple_results]

        parallel_run.close()
        parallel_run.join()


class Initializer(OffspringOperation):
    def __init__(self, conf, log_set):
        OffspringOperation.__init__(self)
        self.conf = conf
        self.box = self.box_generator()
        self.log = log_set

    def initial_mode(self, ope, now_cluster):
        self.log.log_msg += f" - Start Initialization Calculation at Cluster {now_cluster} \n"
        path_cluster = os.path.join(self.conf.home_path, 'Cluster' + str(now_cluster))
        path_gen = os.path.join(path_cluster, 'Gen0')

        iter_list = list(range(self.conf.initial_size))
        iter_list = restart_confirm(path_gen, iter_list)

        os.chdir(path_gen)
        self.parallel_frame(ope, iter_list, path_gen=path_gen)

    def box_generator(self):
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

        atoms = self.conf.stru_min

        cell = atoms.get_cell()
        slab_min_pos = atoms.get_positions()
        zmax = max(slab_min_pos[self.conf.top_num:atoms.get_global_number_of_atoms(), 2]) # The z position of top atom
        height_gap = self.cal_height_gap('Pt')
        min_z = (zmax + 1) / cell[2][2]  # increase 1 A
        d_z = height_gap * self.conf.height_ratio / cell[2][2]  # One layer

        if min_z + d_z > 0.8:
            raise Warning("The initialized box is too high, maybe not vacuum layer.")

        box = [d_z, min_z]
        return box

    def cal_height_gap(self, label):
        atoms = self.conf.stru_min

        tmp = [atom.position[2] for atom in atoms if atom.symbol == label]
        for i in range(len(tmp)):
            if tmp[0] - tmp[i] > self.conf.blmin[(atomic_numbers[label], atomic_numbers[label])]:
                return tmp[0] - tmp[i]

    def run(self):
        self.log.save_log(initial=True)

        if self.conf.cluster_mode == 'Single':
            self.log.log_msg += 'Initialization with Single Cluster Mode \n'
            self.initial_mode(self.parallel_initializer, 0)

        elif self.conf.cluster_mode == 'Multi':
            self.log.log_msg += 'Initialization with Multi-Cluster Mode \n'
            for now_cluster in range(self.conf.num_cluster):
                self.initial_mode(self.parallel_initializer, now_cluster)

        self.log.create_log_each_gen()

        self.log.log_msg += 'Initialization Finished \n'
        self.log.save_log()

    def parallel_initializer(self, item, path_gen, _):
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
            shutil.rmtree(subpath)
            self.parallel_initializer(item, path_gen, _)
        else:
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

            if not atoms_too_close(top, self.conf.blmin, self.conf.adsorp_atom):
                stru_not_good = atoms_too_close_two_sets(top, down, self.conf.blmin)

        slab_temp = top + down
        return slab_temp


class FollowingGen(OffspringOperation):
    def __init__(self, gen, conf, log_set, samp):
        OffspringOperation.__init__(self)
        self.gen = gen

        self.log = log_set

        self.conf = conf
        self.conf.follow_initial()

        self.samp = samp

        self.candidate_list = multiprocess.Manager().list()

        self.parent_ga_list = multiprocess.Manager().list()
        self.parent_de_list = multiprocess.Manager().list()

        self.for_calc_list = None

        self.op_selector = OperationSelector([self.conf.ga_de_ratio * 0.8, self.conf.ga_de_ratio * 0.2,  1-self.conf.ga_de_ratio],
                                             [self.ga_op_1, self.ga_op_2, self.de_op])

        self.iter_list = list()

    def ga_op_1(self, now_cluster, max_count=800):
        method = 'ga_1'

        list_for_cross = self.log.rank_each_cluster[now_cluster]
        path_cluster = os.path.join(self.conf.home_path, 'Cluster' + str(now_cluster))

        too_close = True
        count = 0
        while too_close:
            count += 1
            if count > max_count:
                break

            parent_array = return_parent_number(self.log, self.conf, now_cluster, num=2)
            p1 = parent_array[0]
            p2 = parent_array[1]

            parent = [p1, p2]
            parent.sort()

            if parent not in self.parent_ga_list:
                a1 = read_structure(list_for_cross[p1], path_cluster)
                a2 = read_structure(list_for_cross[p2], path_cluster)

                slab = a1[self.conf.num_atom:]

                pairing = modified_CutAndSplicePairing(slab, self.conf.num_atom, self.conf.blmin)
                stru_temp = pairing.cross(a1, a2)

                if stru_temp is not None:
                    if random.randrange(0, 100) < self.conf.mutate_rate:
                        stru_mutate, mutate_type = random.choice([rattle(stru_temp, self.conf.blmin, self.conf.num_atom),
                                                     permutation(stru_temp, self.conf.blmin, self.conf.num_atom),
                                                     mirror(stru_temp, self.conf.blmin, self.conf.num_atom)])
                        if stru_mutate is not None:
                            stru_temp = stru_mutate
                            method = 'ga_1_with_' + mutate_type

                    stru_temp.info.update({'Parent': parent, 'Method': method})
                    return stru_temp

    def ga_op_2(self, now_cluster, max_count=1000):
        method = 'ga_2'
        stru_temp = copy.deepcopy(self.conf.stru)
        stru_temp_pos = stru_temp.get_scaled_positions()

        list_for_cross = self.log.rank_each_cluster[now_cluster]
        path_cluster = os.path.join(self.conf.home_path, 'Cluster' + str(now_cluster))

        too_close = True
        count = 0
        while too_close:
            count += 1
            if count > max_count:
                break

            parent_array = return_parent_number(self.log, self.conf, now_cluster, num=2)
            p1 = parent_array[0]
            p2 = parent_array[1]

            parent = [p1, p2]
            parent.sort()

            if parent not in self.parent_ga_list:
                pos_p1 = read_structure(list_for_cross[p1], path_cluster, return_pos=True)
                pos_p2 = read_structure(list_for_cross[p2], path_cluster, return_pos=True)
                new_stru_pos = crossover(pos_p1, pos_p2, stru_temp_pos, self.conf.num_elem_atom)

                if new_stru_pos is not None:
                    stru_temp.set_scaled_positions(new_stru_pos)
                    too_close = atoms_too_close(stru_temp[:self.conf.num_atom], self.conf.blmin)

                    if not too_close:
                        if random.randrange(0, 100) < self.conf.mutate_rate:
                            stru_mutate, mutate_type = random.choice([rattle(stru_temp, self.conf.blmin, self.conf.num_atom),
                                                         permutation(stru_temp, self.conf.blmin, self.conf.num_atom),
                                                         mirror(stru_temp, self.conf.blmin, self.conf.num_atom)])
                            if stru_mutate is not None:
                                stru_temp = stru_mutate
                                method = 'ga_2_with_' + mutate_type

                        stru_temp.info.update({'Parent': parent, 'Method': method})
                        return stru_temp

    def de_op(self, now_cluster, max_count=50):
        too_close = True
        count = 0

        while too_close:
            count += 1
            if count > max_count:
                break

            de_method = random.choice([de_rand_1, de_best_1, de_rand_to_best_1])
            stru_temp, parent, method = de_method(self.log, self.conf, now_cluster, self.parent_de_list)

            if stru_temp is not None:
                if not atoms_too_close_two_sets(stru_temp[:self.conf.num_atom], stru_temp[self.conf.num_atom:],
                                                self.conf.blmin):
                    if not atoms_too_close(stru_temp, self.conf.blmin):
                        stru_temp.info.update({'Parent': parent, 'Method': method})
                        return stru_temp

    def random_ope_sampling(self, now_cluster, output_single_structure=False):
        accept = False
        while accept is False:
            op = self.op_selector.get_operator()
            new_stru = op(now_cluster)

            if new_stru is not None:
                # if not sampling_similar(new_stru, self.conf, self.log, now_cluster):
                if 'ga' in new_stru.info['Method']:
                    self.parent_ga_list.append(new_stru.info['Parent'])
                elif 'de' in new_stru.info['Method']:
                    self.parent_de_list.append(new_stru.info['Parent'])

                accept = True

        if output_single_structure is True:
            return new_stru
        else:
            self.candidate_list.append(new_stru)

    def offspring_mode(self, ope, now_cluster):
        assert len(self.iter_list) == len(self.for_calc_list)
        path_cluster = os.path.join(self.conf.home_path, 'Cluster' + str(now_cluster))
        path_gen = os.path.join(path_cluster, f'Gen{self.gen}')

        os.chdir(path_gen)
        self.parallel_frame(ope, self.iter_list, path_gen=path_gen, now_cluster=now_cluster)

    def parallel_offspring(self, item, path_gen, now_cluster):
        '''
        这是用来计算的

        :param item:
        :param path_gen:
        :param now_cluster:
        :return:
        '''
        subpath = os.path.join(path_gen, str(item))
        if not os.path.exists(str(subpath)):
            os.mkdir(str(subpath))

        os.chdir(subpath)

        stru = self.for_calc_list[self.iter_list.index(item)]
        stru.calc = self.conf.calc

        try:
            stru.get_potential_energy()
        except ValueError or KeyError:
            self.for_calc_list[self.iter_list.index(item)] = self.random_ope_sampling(now_cluster,
                                                                                      output_single_structure=True)
            shutil.rmtree(subpath)
            self.parallel_offspring(item, path_gen, now_cluster)
        else:
            stru.info.update({'Calculated_Env': stru.calc.results['energy']})
            self.record_stru_info(stru, subpath, now_cluster)

            # ase.io.write(os.path.join(subpath, "CONTCAR"), stru, format='vasp')
            ase.io.write(os.path.join(subpath, f"output_{self.gen}_" + str(item) + '.cif'), stru)
            ga_site = os.path.join(path_gen, "GAlog.txt")
            os.system(f"echo {self.gen} {item} {stru.calc.results['energy']} >> {ga_site}")

    def record_stru_info(self, stru, subpath, now_cluster):
        path_cluster = os.path.join(self.conf.home_path, 'Cluster' + str(now_cluster))
        list_for_comm = self.log.rank_each_cluster[now_cluster]

        stru_path = list()
        for item in stru.info['Parent']:
            path = os.path.join(path_cluster, "Gen" + str(int(list_for_comm[item][0])),
                                str(int(list_for_comm[item][1])), "CONTCAR")
            stru_path.append(path)

        stru.info['Parent'] = stru_path

        with open(os.path.join(subpath, "info.json"), 'w') as json_file:
            json.dump(stru.info, json_file, indent=4, separators=(',', ':'))

    def samp_list_generator(self, now_clu):
        samp_list = range(len(self.iter_list) * self.conf.samp_ratio)
        path_clu = os.path.join(self.conf.home_path, 'Cluster' + str(now_clu))
        save_file = os.path.join(path_clu, f'Gen{self.conf.gen}', 'samp_candidate.json')

        if os.path.exists(save_file):
            pickle_data = open(save_file, 'rb')
            self.candidate_list = pickle.load(pickle_data)
            # assert len(samp_list) == len(self.candidate_list) # 如果算了一部分，断点重开就会不一样
        else:
            self.parallel_frame(self.random_ope_sampling, samp_list, now_cluster=now_clu, task='sampling')
            with open(save_file, 'wb') as file_handle:
                pickle.dump(list(self.candidate_list), file_handle)

    def run(self):
        if self.conf.cluster_mode == 'Single':  # TODO
            self.log.log_msg += 'Initialization with Single Cluster Mode \n'

        elif self.conf.cluster_mode == 'Multi':
            for now_cluster in range(self.conf.num_cluster):
                path_gen = os.path.join(self.conf.home_path, 'Cluster' + str(now_cluster), 'Gen' + str(self.gen))
                self.iter_list = list(range(self.conf.num_pop))
                self.iter_list = restart_confirm(path_gen, self.iter_list)

                if len(self.iter_list) == 0:
                    continue

                if 0 in self.iter_list:
                    exchange(self.gen, now_cluster, self.conf, self.log)
                    self.iter_list.remove(0)

                self.samp_list_generator(now_cluster)
                self.samp.update(now_cluster)  # Update training dataset before sampling for next generation.
                self.for_calc_list = self.samp.sample(self.candidate_list, len(self.iter_list), now_cluster=now_cluster)
                self.offspring_mode(self.parallel_offspring, now_cluster)

            if self.conf.gen % self.conf.ml_interval == 0:
                self.samp.gpr_model.train()

        self.samp.error_stat(self.conf, self.gen)
        self.log.stru_generation_method_stat(self.conf, self.gen)
        self.log.create_log_each_gen(gen=self.gen)


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
        return self.oplist[to_use]

def restart_confirm(path_gen, iter_list):
    if not os.path.exists(path_gen):
        os.makedirs(path_gen)
    else:
        previous_cal_list = list()

        if os.path.exists(os.path.join(path_gen, 'GAlog.txt')):
            file = open(os.path.join(path_gen, 'GAlog.txt'), 'r')
            previous_cal_list = [int(x.split()[1]) for x in file]

        for i in list(set(previous_cal_list)):
            iter_list.remove(i)

        print(f"Restart: This Cluster Already Calculate {len(previous_cal_list)} Structures")

    return iter_list


def return_parent_number(log_set, conf, now_cluster, num=2,
                         return_stru=False, return_best_stru=False):
    path_cluster = os.path.join(conf.home_path, 'Cluster' + str(now_cluster))

    random_list = list(range(conf.num_fit))
    parent_array = np.zeros(num, dtype=np.int64)
    list_for_comm = log_set.rank_each_cluster[now_cluster]
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


def read_structure(stru_info, home_path, return_pos=False):
    stru_path = os.path.join(home_path, "Gen" + str(int(stru_info[0])), str(int(stru_info[1])), "CONTCAR")
    stru = ase.io.read(stru_path, format='vasp')
    stru = stru[np.argsort(-stru.get_positions()[..., 2])]

    if return_pos:
        return stru.get_scaled_positions()

    return stru

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

class modified_CutAndSplicePairing(CutAndSplicePairing):
    def __init__(self, slab, n_top, blmin):
        CutAndSplicePairing.__init__(self, slab, n_top, blmin)

    def cross(self, a1, a2, maxcount=150):
        """Crosses the two atoms objects and returns one"""

        if len(a1) != len(self.slab) + self.n_top:
            raise ValueError('Wrong size of structure to optimize')
        if len(a1) != len(a2):
            raise ValueError('The two structures do not have the same length')

        N = self.n_top
        # Only consider the atoms to optimize
        a1 = a1[:N]
        a2 = a2[:N]

        a1 = ase.build.sort(a1)
        a2 = ase.build.sort(a2)

        if not np.array_equal(a1.numbers, a2.numbers):
            err = 'Trying to pair two structures with different stoichiometry'
            raise ValueError(err)

        if self.use_tags and not np.array_equal(a1.get_tags(), a2.get_tags()):
            err = 'Trying to pair two structures with different tags'
            raise ValueError(err)

        cell1 = a1.get_cell()
        cell2 = a2.get_cell()
        for i in range(self.number_of_variable_cell_vectors, 3):
            err = 'Unit cells are supposed to be identical in direction %d'
            assert np.allclose(cell1[i], cell2[i]), (err % i, cell1, cell2)

        invalid = True
        counter = 0
        maxcount = maxcount
        a1_copy = a1.copy()
        a2_copy = a2.copy()

        # Run until a valid pairing is made or maxcount pairings are tested.
        while invalid and counter < maxcount:
            counter += 1

            newcell = self.generate_unit_cell(cell1, cell2)
            if newcell is None:
                # No valid unit cell could be generated.
                # This strongly suggests that it is near-impossible
                # to generate one from these parent cells and it is
                # better to abort now.
                break

            # Choose direction of cutting plane normal
            if self.number_of_variable_cell_vectors == 0:
                # Will be generated entirely at random
                theta = np.pi * self.rng.random()
                phi = 2. * np.pi * self.rng.random()
                cut_n = np.array([np.cos(phi) * np.sin(theta),
                                  np.sin(phi) * np.sin(theta), np.cos(theta)])
            else:
                # Pick one of the 'variable' cell vectors
                cut_n = self.rng.choice(self.number_of_variable_cell_vectors)

            # Randomly translate parent structures
            for a_copy, a in zip([a1_copy, a2_copy], [a1, a2]):
                a_copy.set_positions(a.get_positions())

                cell = a_copy.get_cell()
                for i in range(self.number_of_variable_cell_vectors):
                    r = self.rng.random()
                    cond1 = i == cut_n and r < self.p1
                    cond2 = i != cut_n and r < self.p2
                    if cond1 or cond2:
                        a_copy.positions += self.rng.random() * cell[i]

                if self.use_tags:
                    # For correct determination of the center-
                    # of-position of the multi-atom blocks,
                    # we need to group their constituent atoms
                    # together
                    gather_atoms_by_tag(a_copy)
                else:
                    a_copy.wrap()

            # Generate the cutting point in scaled coordinates
            cosp1 = np.average(a1_copy.get_scaled_positions(), axis=0)
            cosp2 = np.average(a2_copy.get_scaled_positions(), axis=0)
            cut_p = np.zeros((1, 3))
            for i in range(3):
                if i < self.number_of_variable_cell_vectors:
                    cut_p[0, i] = self.rng.random()
                else:
                    cut_p[0, i] = 0.5 * (cosp1[i] + cosp2[i])

            # Perform the pairing:
            child = self._get_pairing(a1_copy, a2_copy, cut_p, cut_n, newcell)
            if child is None:
                continue

            # Verify whether the atoms are too close or not:
            if atoms_too_close(child, self.blmin, use_tags=self.use_tags):
                continue

            if self.test_dist_to_slab and len(self.slab) > 0:
                if atoms_too_close_two_sets(child, self.slab, self.blmin):
                    continue

            # Passed all the tests
            child = child + self.slab
            child.set_cell(newcell, scale_atoms=False)
            child.wrap()
            return child

        return None

def de_rand_1(log_set, conf, now_cluster, parent_de_list):
    final = None
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

    return final, parent, de_rand_1.__name__


def de_best_1(log_set, conf, now_cluster, parent_de_list):
    final = None

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

    return final, parent, de_best_1.__name__


def de_rand_to_best_1(log_set, conf, now_cluster, parent_de_list):
    final = None

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

    return final, parent, de_rand_to_best_1.__name__
