import os
import shutil
import subprocess

import ase.io
from ase.calculators import calculator
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

    def calculate(self,
                  atoms=None,
                  properties=['energy'],
                  system_changes=['positions']):

        self.write_input(atoms)

        self._run(command=self.lasp_command,
                  directory=self.directory)

        lasp_error = self.update_atoms()

        if lasp_error:
            raise ValueError('Input Structure Cause Problem in {}'
                             .format(os.path.abspath(self.directory)))

        self.read_energy()

        if self.clear_output:
            self._clear_output()

    # Methods for reading information from LASP output files:
    def read_energy(self):
        """Method to read energy from best.arc file."""
        with open('best.arc', 'r') as x:
            energy_line = x.readlines()[2]
            energy = energy_line.split()[3]
        self.results['energy'] = energy

    def update_atoms(self):
        """Update the atoms object with new positions and cell"""
        try:
            self.atoms = ase.io.read('best.arc', format='dmol-arc')
        except StopIteration:
            lasp_error = True
            return lasp_error

    def write_input(self, atoms):
        ase.io.write('input.arc', atoms, format='dmol-arc')
        shutil.copy(self.lasp_in, os.getcwd())
        shutil.copy(self.pot_file, os.getcwd())

    def _run(self, command=None, out=None, directory=None):
        """Method to explicitly execute LASP"""
        if command is None:
            command = self.command
        if directory is None:
            directory = self.directory

        subprocess.call(command, shell=True, stdout=out, cwd=directory)

    def _clear_output(self):
        delete_list = ['all.arc', 'allfor.arc', 'allkeys.log', 'allstr.arc',
                       'Badstr.arc', 'lasp.in', 'lasp.out', 'SSWtraj']

        file_list = os.listdir(self.directory)

        for item in file_list:
            if item in delete_list or (item.endswith(".pot")):
                os.remove(item)
