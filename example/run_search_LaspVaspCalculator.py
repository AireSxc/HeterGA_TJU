import sys

sys.path.append('/hpctmp/e0444250/TJU_HeteroGA/heteroga')

import numpy as np
from ase.constraints import FixAtoms
from ase.calculators.vasp import Vasp

from calculator import LaspVaspCalculator
from setting import ReadSetandCalc
from opt import HeterogaSurface

vasp_calc = Vasp()
vasp_calc.read_incar('/hpctmp/e0444250/TJU_HeteroGA_0.75PtO/LASP_VASP_cal/cal_set/INCAR')
vasp_calc.read_kpoints('/hpctmp/e0444250/TJU_HeteroGA_0.75PtO/LASP_VASP_cal/cal_set/KPOINTS')
vasp_calc.read_potcar('/hpctmp/e0444250/TJU_HeteroGA_0.75PtO/LASP_VASP_cal/cal_set/POTCAR')

def extra_pcs(atoms):
    atoms = atoms[np.argsort(-atoms.get_positions()[..., 2])]
    del atoms[-32:]
    c = FixAtoms(indices=[*range(len(atoms) - 16, len(atoms))])
    atoms.set_constraint(c)
    return atoms

lasp_calc = LaspVaspCalculator(
    lasp_command='mpirun -np 8 /home/svu/e0444250/install/LASP_FULL_pro2_0_1_intel12_sss/Src/lasp',
    lasp_in='/hpctmp/e0444250/TJU_GongGA/heteroga/example/lasp.in',
    pot_file='/hpctmp/e0444250/TJU_GongGA/heteroga/example/PtO.pot',
    vasp_calc=vasp_calc,
    extra_pcs=extra_pcs,
    vasp_cmd='mpirun -np 12 /home/svu/e0444250/install/vasp544/vasp_std')

conf = ReadSetandCalc(file='config.yaml', calc=lasp_calc)

HeterogaSurface(conf=conf).run()
