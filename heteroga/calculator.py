import os
import shutil
import subprocess

import ase.io
import numpy as np
from ase.calculators.calculator import Calculator

class LaspCalculator(Calculator):
    implemented_properties = ['energy']
    default_parameters = {}

    def __init__(self,
                 lasp_command=None,
                 lasp_in=None,
                 pot_file=None,
                 directory='.',
                 clear_output=True,
                 **kwargs):
        Calculator.__init__(self, **kwargs)

        self.directory = directory

        if lasp_command is None:
            raise ValueError('LASP command was not found')
        else:
            self.lasp_command = lasp_command

        if lasp_in is None or not os.path.exists(lasp_in):
            raise ValueError('LASP input file (.in) was not found')
        else:
            self.lasp_in = lasp_in

        if pot_file is None or not os.path.exists(pot_file):
            raise ValueError('NN potential file (.pot) was not found')
        else:
            self.pot_file = pot_file

        self.clear_output = clear_output
        self.delete_list = ['all.arc', 'allfor.arc', 'allkeys.log', 'allstr.arc',
                            'Badstr.arc', 'lasp.in', 'lasp.out', 'SSWtraj']

    def calculate(self,
                  atoms=None,
                  properties=['energy'],
                  system_changes=['positions']):

        self.write_input(atoms)

        self.run()

        self.update_atoms_and_energy()

        if self.clear_output:
            self._clear_output()

    def update_atoms_and_energy(self):
        """Update the atoms object with new positions and cell"""
        if os.path.exists('ExceedSym.arc'):
            raise ValueError('Input Structure Cause Problem in {}'.format(os.path.abspath(self.directory)))
        else:
            try:
                self.atoms = ase.io.read('best.arc', format='dmol-arc')
                with open('best.arc', 'r') as x:
                    energy_line = x.readlines()[2]
                    energy = energy_line.split()[3]
            except StopIteration:
                self.atoms = ase.io.read('all.arc', format='dmol-arc')
                with open('all.arc', 'r') as x:
                    energy_line = x.readlines()[2]
                    energy = energy_line.split()[3]

            self.results['energy'] = energy

    def write_input(self, atoms):
        ase.io.write('input.arc', atoms, format='dmol-arc')
        shutil.copy(self.lasp_in, os.getcwd())
        shutil.copy(self.pot_file, os.getcwd())

    def run(self, command=None, out=None, directory=None):
        """Method to explicitly execute LASP"""
        if command is None:
            command = self.lasp_command

        if directory is None:
            directory = self.directory

        subprocess.call(command, shell=True, stdout=out, cwd=directory)

    def _clear_output(self):
        file_list = os.listdir(self.directory)

        for item in file_list:
            if item in self.delete_list or (item.endswith(".pot")):
                os.remove(item)


class LaspVaspCalculator(LaspCalculator):
    implemented_properties = ['energy']
    default_parameters = {}

    def __init__(self,
                 vasp_calc=None,
                 extra_pcs=None,
                 vasp_cmd=None,
                 **kwargs):

        LaspCalculator.__init__(self, **kwargs)

        self.vasp_calc = vasp_calc
        self.extra_pcs = extra_pcs
        self.vasp_cmd = vasp_cmd
        self.delete_list = ['all.arc', 'allfor.arc', 'allkeys.log', 'allstr.arc',
                            'Badstr.arc', 'lasp.in', 'lasp.out', 'SSWtraj',
                            'REPORT', 'CHGCAR', 'CHG', 'DOSCAR', 'EIGENVAL', 'IBZKPT', 'PCDAT', 'PROCAR',
                            'WAVECAR', 'XDATCAR', 'vasprun.xml', 'FORCECAR']

    def update_atoms_and_energy(self):
        """Update the atoms object with new positions and cell"""
        if os.path.exists('ExceedSym.arc'):
            raise ValueError('Input Structure Cause Problem in {}'.format(os.path.abspath(self.directory)))
        else:
            try:
                self.atoms = ase.io.read('best.arc', format='dmol-arc')
            except StopIteration:
                self.atoms = ase.io.read('all.arc', format='dmol-arc')

        vasp_atom = self.extra_pcs(self.atoms)
        vasp_atom = vasp_atom[np.argsort(-vasp_atom.get_positions()[..., 2])]
        self.vasp_calc.write_input(vasp_atom)

        subprocess.call(self.vasp_cmd, shell=True, stdout=subprocess.DEVNULL)

        energy_free, energy_zero = self.vasp_calc.read_energy()
        self.results['energy'] = energy_zero
