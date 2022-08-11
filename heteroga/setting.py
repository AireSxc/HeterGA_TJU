import os
from functools import reduce

import ase.io
import numpy as np
import yaml
from ase.calculators.calculator import Calculator
from ase.ga.utilities import closest_distances_generator

from utilities import elem_atom_generator, rearrange_order

class ReadSetandCalc:
    def __init__(self,
                 file='config.yaml',
                 calc=None):

        conf = yaml.load(open(file, 'r'), Loader=yaml.FullLoader)

        # Whole Program setting

        try:
            conf['ene_shield']
        except KeyError:
            self.ene_shield = float("-inf")
        else:
            self.ene_shield = conf['ene_shield']

        try:
            conf['logfile']
        except KeyError:
            self.logfile = 'search.log'
        else:
            self.logfile = conf['logfile']

        self.home_path = os.getcwd()
        self.data_save_file = 'features_ene.pkl'
        self.model_save_file = 'model.pth'

        try:
            conf['adsorp_atom']
        except KeyError:
            self.adsorp_atom = [8]
        else:
            self.adsorp_atom = conf['adsorp_atom']

        self.stru_file = conf['stru_file']

        self.stru_min = ase.io.read(os.path.join(self.home_path, self.stru_file))
        self.stru_min = self.stru_min[np.argsort(-self.stru_min.get_positions()[..., 2])]

        all_atom_types = list(set(self.stru_min.numbers))
        self.blmin = closest_distances_generator(all_atom_types, ratio_of_covalent_radii=0.7)

        assert conf['num_cluster'] > 0
        if conf['num_cluster'] == 1:
            self.cluster_mode = 'Single'
        else:
            self.cluster_mode = 'Multi'
            self.num_cluster = conf['num_cluster']

        self.num_gen = conf['num_gen']

        assert calc is not None and isinstance(calc, Calculator)
        self.calc = calc

        # Initializer
        self.num_fit = conf['num_fit']
        self.initial_size = conf['num_fit'] * conf['initial_ratio']

        self.supercell_num = tuple(map(int, conf['supercell_set'].split(', ')))

        self.top_num = conf['top_num_atom_for_optima']
        self.num_elem_atom_min = elem_atom_generator(self.stru_min, conf['top_num_atom_for_optima'])
        self.num_atom_min = sum(self.num_elem_atom_min)

        self.num_elem_atom = [i * reduce(lambda x, y: x * y, self.supercell_num) for i in self.num_elem_atom_min]
        self.num_atom = sum(self.num_elem_atom)

        top = rearrange_order(self.stru_min[:self.num_atom_min] * self.supercell_num)
        down = self.stru_min[self.num_atom_min:] * self.supercell_num
        self.stru = top + down

        self.height_ratio = conf['height_ratio']

        self.delete_num = conf['delete_num']  # TODO LIST
        self.initial_mode = conf['initial_mode']

        self.gen = 0

        # Sampling

        self.sampling = conf['sampling_method']
        self.samp_ratio = conf['sampling_ratio']

        # Following generation

        try:
            conf['mutate_rate']
        except KeyError:
            self.mutate_rate = 1.0 * 100
        else:
            self.mutate_rate = conf['mutate_rate']

        self.num_pop = conf['num_pop']

        try:
            conf['ga_de_ratio']
        except KeyError:
            self.ga_de_ratio = 1
        else:
            self.ga_de_ratio = conf['ga_de_ratio']

        self.ml_interval = conf['ml_interval']
        self.e_pt = -379.70

        try:
            conf['ene_shield']
        except KeyError:
            self.ene_shield = 0.92
        else:
            self.ene_shield = conf['ene_shield']

        self.training_iter = conf['training_iter']

        try:
            conf['max_data_size']
        except KeyError:
            self.max_data_size = 2500
        else:
            self.max_data_size = conf['max_data_size']

    def follow_initial(self):
        self.stru = ase.io.read(os.path.join(self.home_path, 'Cluster0', 'Gen0', '0', 'CONTCAR'))

