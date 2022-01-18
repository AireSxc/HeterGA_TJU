import os
from functools import reduce

import ase.io
import numpy as np
import yaml
from ase.calculators.calculator import Calculator

from geometry_check import closest_distances_generator
from utilities import elem_atom_generator


class ReadSetandCalc:
    def __init__(self, file='config.yaml', calc=None):

        conf = yaml.load(open(file, 'r'), Loader=yaml.FullLoader)

        # Whole Program setting
        self.home_path = os.getcwd()

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
        self.blmin = closest_distances_generator(self.adsorp_atom, all_atom_types, ratio_of_covalent_radii=0.7)

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

        self.num_elem_atom = [i * reduce(lambda x, y: x * y, self.supercell_num) for i in self.num_elem_atom_min]
        self.num_atom = sum(self.num_elem_atom)

        self.height_ratio = conf['height_ratio']

        self.delete_num = conf['delete_num']  # TODO LIST
        self.initial_mode = conf['initial_mode']

        # Sampling

        self.sampling = conf['sampling_method']
        self.samp_ratio = conf['sampling_ratio']

        # Following generation

        self.slab = None

        try:
            conf['mutate_rate']
        except KeyError:
            self.mutate_rate = 1.0 * 100
        else:
            self.mutate_rate = conf['mutate_rate']

    def follow_initial(self):
        self.slab = ase.io.read(os.path.join(home_path, 'Cluster0', 'Gen0', '0', 'CONTCAR'))
