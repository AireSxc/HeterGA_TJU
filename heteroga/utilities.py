import os
import numpy as np

import ase.io

def get_sorted_dist_list(atoms, mic=False):
    """ Utility method used to calculate the sorted distance list
        describing the cluster in atoms.
        mic: Determines if distances are calculated
        using the minimum image convention"""
    numbers = atoms.numbers
    unique_types = set(numbers)
    pair_cor = dict()
    for n in unique_types:
        i_un = [i for i in range(len(atoms)) if atoms[i].number == n]
        d = []
        for i, n1 in enumerate(i_un):
            for n2 in i_un[i + 1:]:
                d.append(atoms.get_distance(n1, n2, mic))
        d.sort()
        pair_cor[n] = np.array(d)
    return pair_cor


def read_structure(stru_info, home_path):
    path = home_path + "/Gen" + str(int(stru_info[0])) + "/" + str(int(stru_info[1])) + "/CONTCAR"
    slab = ase.io.read(path, format='vasp')
    return slab.get_scaled_positions()


def elem_atom_generator(atoms, top_num):
    tmp = []
    for v in np.unique(atoms[:top_num].numbers):
        tmp.append(np.sum(atoms[:top_num].numbers == v))
    return tmp


def split_based(a, val):
    # https://stackoverflow.com/questions/40835066/delete-element-from-multi-dimensional-numpy-array-by-value
    mask = a != val
    p = np.split(a[mask], mask.sum(1)[:-1].cumsum())
    out = np.array(list(map(list, p)))
    return out

def rearrange_order(top_slab):
    symbols = top_slab.get_chemical_symbols()
    pos = top_slab.get_positions()

    symbols_set = []
    [symbols_set.append(i) for i in symbols if i not in symbols_set]

    new_symbols = []
    new_pos = []

    for h in range(len(symbols_set)):
        for i in range(len(pos)):
            if symbols[i] == symbols_set[h]:
                new_symbols.append(symbols_set[h])
                new_pos.append(pos[i])

    top_slab.set_chemical_symbols(new_symbols)
    top_slab.set_positions(new_pos)
    return top_slab


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
