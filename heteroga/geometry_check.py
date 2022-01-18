#!/usr/bin/env python3.6

import itertools
import numpy as np

from ase.data import covalent_radii
from scipy.spatial.distance import cdist

from utilities import get_sorted_dist_list, split_based


def closest_distances_generator(adsorption_atom, atom_numbers, ratio_of_covalent_radii):  # From ASE
    """ Generates the blmin dict used across the GA.
        The distances are based on the covalent radii of the atoms.
    """
    cr = covalent_radii
    ratio = ratio_of_covalent_radii

    blmin = dict()

    for i in atom_numbers:
        blmin[(i, i)] = cr[i] * 2 * ratio
        for j in atom_numbers:
            if i == j:
                continue
            if (i, j) in blmin.keys():
                continue
            blmin[(i, j)] = blmin[(j, i)] = ratio * (cr[i] + cr[j])

    if adsorption_atom is not None:
        for k in adsorption_atom:
            blmin[(k, k)] = cr[k] * 2 * 1.25

    return blmin


def atoms_too_close(atoms, bl, adsorp_atom=None):  # From ASE
    """ Checks if any atoms in a are too close, as defined by
        the distances in the bl dictionary. """
    a = atoms.copy()
    pbc = a.get_pbc()
    cell = a.get_cell()
    num = a.get_atomic_numbers()
    pos = a.get_positions()
    unique_types = sorted(list(set(num)))

    distance_inside = cdist(pos, pos)
    iterator = itertools.combinations_with_replacement(unique_types, 2)
    for type1, type2 in iterator:
        x1 = np.where(num == type1)
        x2 = np.where(num == type2)
        dis_nozero = split_based(distance_inside[x1].T[x2], 0)

        if np.min(dis_nozero) < bl[(type1, type2)]:
            return True

        if adsorp_atom in [type1, type2] and type1 != type2:
            for item in dis_nozero:
                if np.min(item) > (covalent_radii[type1] + covalent_radii[type2]) * 1.2:
                    return True

    neighbours = []
    for i in range(3):
        if pbc[i]:
            neighbours.append([-1, 0, 1])
        else:
            neighbours.append([0])

    for nx, ny, nz in itertools.product(*neighbours):
        displacement = np.dot(cell.T, np.array([nx, ny, nz]).T)
        pos_new = pos + displacement
        distances = cdist(pos, pos_new)

        if nx == 0 and ny == 0 and nz == 0:
            distances += 1e2 * np.identity(len(a))

        iterator = itertools.combinations_with_replacement(unique_types, 2)
        for type1, type2 in iterator:
            x1 = np.where(num == type1)
            x2 = np.where(num == type2)
            if np.min(distances[x1].T[x2]) < bl[(type1, type2)]:
                return True
    return False


def cum_diff(a1, a2):
    p1 = get_sorted_dist_list(a1)
    p2 = get_sorted_dist_list(a2)
    numbers = a1.numbers
    total_cum_diff = 0.
    for n in p1.keys():
        c1 = p1[n]
        c2 = p2[n]
        assert len(c1) == len(c2)
        if len(c1) == 0:
            continue
        t_size = np.sum(c1)
        d = np.abs(c1 - c2)
        cum_diff_sum = np.sum(d)
        ntype = float(sum([i == n for i in numbers]))
        total_cum_diff += cum_diff_sum / t_size * ntype / float(len(numbers))
    return total_cum_diff


def looks_like(a1, a2, num_atom, delta_de, pair_cor_cum_diff):
    """ Return if structure a1 or a2 are similar or not. """
    # first we check the energy criteria
    de = abs(a1[1] - a2[1])
    if de >= delta_de:
        return False

    # then we check the structure
    total_cum_diff = cum_diff(a1[0][:num_atom], a2[0][:num_atom])

    return total_cum_diff < pair_cor_cum_diff


def atoms_too_close_two_sets(a, b, bl):
    """ Checks if any atoms in a are too close to an atom in b,
        as defined by the bl dictionary. """
    tot = a + b
    num = tot.numbers
    for i in range(len(a)):
        for j in range(len(a), len(tot)):
            if tot.get_distance(i, j, True) < bl[(num[i], num[j])]:
                return True
    return False
