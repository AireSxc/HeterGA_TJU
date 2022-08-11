import math
import random

import numpy as np
from ase import Atoms
from ase.ga.utilities import get_rotation_matrix, gather_atoms_by_tag

from geometry_check import atoms_too_close, atoms_too_close_two_sets

mc = 500

def rattle(old_atoms, blmin, num_atom, maxcount=mc, rattle_strength=0.8, rattle_prop=0.4, test_dist_to_slab=True, use_tags=False):
    """ Does the actual mutation. """
    n = len(old_atoms) if num_atom is None else num_atom
    atoms = old_atoms[:n]
    slab = old_atoms[n:]
    tags = atoms.get_tags() if use_tags else np.arange(n)
    pos_ref = atoms.get_positions()
    num = atoms.get_atomic_numbers()
    cell = atoms.get_cell()
    pbc = atoms.get_pbc()
    st = 2. * rattle_strength

    count = 0
    too_close = True; mutant = None
    while too_close and count < maxcount:
        count += 1
        pos = pos_ref.copy()
        ok = False
        for tag in np.unique(tags):
            select = np.where(tags == tag)
            if np.random.random() < rattle_prop:
                ok = True
                r = np.random.random(3)
                pos[select] += st * (r - 0.5)

        if not ok:
            # Nothing got rattled
            continue

        top = Atoms(num, positions=pos, cell=cell, pbc=pbc, tags=tags)
        too_close = atoms_too_close(top, blmin)
        if not too_close and test_dist_to_slab:
            too_close = atoms_too_close_two_sets(top, slab, blmin)
        mutant = top + slab

    if count == maxcount:
        return None, rattle.__name__

    return mutant, rattle.__name__


def permutation(old_atoms, blmin, num_atom, maxcount=mc, probability=0.4, test_dist_to_slab=True, use_tags=False):
    """ Does the actual mutation. """
    n = len(old_atoms) if num_atom is None else num_atom
    atoms = old_atoms[:n]
    slab = old_atoms[n:]
    if use_tags:
        gather_atoms_by_tag(atoms)
    tags = atoms.get_tags() if use_tags else np.arange(n)
    pos_ref = atoms.get_positions()
    num = atoms.get_atomic_numbers()
    cell = atoms.get_cell()
    pbc = atoms.get_pbc()
    symbols = atoms.get_chemical_symbols()

    unique_tags = np.unique(tags)
    n = len(unique_tags)
    swaps = int(np.ceil(n * probability / 2.))

    sym = []
    for tag in unique_tags:
        indices = np.where(tags == tag)[0]
        s = ''.join([symbols[j] for j in indices])
        sym.append(s)

    assert len(np.unique(sym)) > 1, \
        'Permutations with one atom (or molecule) type is not valid'

    count = 0
    too_close = True; mutant = None
    while too_close and count < maxcount:
        count += 1
        pos = pos_ref.copy()
        for _ in range(swaps):
            i = j = 0
            while sym[i] == sym[j]:
                i = np.random.randint(0, high=n)
                j = np.random.randint(0, high=n)
            ind1 = np.where(tags == i)
            ind2 = np.where(tags == j)
            cop1 = np.mean(pos[ind1], axis=0)
            cop2 = np.mean(pos[ind2], axis=0)
            pos[ind1] += cop2 - cop1
            pos[ind2] += cop1 - cop2

        top = Atoms(num, positions=pos, cell=cell, pbc=pbc, tags=tags)
        if blmin is None:
            too_close = False
        else:
            too_close = atoms_too_close(top, blmin)
            if not too_close and test_dist_to_slab:
                too_close = atoms_too_close_two_sets(top, slab, blmin)
        mutant = top + slab

    if count == maxcount:
        return None, permutation.__name__

    return mutant, permutation.__name__


def mirror(old_atoms, blmin, num_atom, n_tries=mc, reflect=False):
    """ Do the mutation of the atoms input. """

    tc = True; tot = None
    n = len(old_atoms) if num_atom is None else num_atom
    top = old_atoms[:n]; slab = old_atoms[n:]
    num = top.numbers
    unique_types = list(set(num))
    nu = dict()
    for u in unique_types:
        nu[u] = sum(num == u)

    counter = 0
    changed = False

    while tc and counter < n_tries:
        counter += 1
        cand = top.copy()
        pos = cand.get_positions()

        cm = np.average(top.get_positions(), axis=0)

        # first select a randomly oriented cutting plane
        theta = math.pi * random.random()
        phi = 2. * math.pi * random.random()
        n = (math.cos(phi) * math.sin(theta), math.sin(phi) * math.sin(theta), math.cos(theta))
        n = np.array(n)

        # Calculate all atoms signed distance to the cutting plane
        d_list = []
        for (i, p) in enumerate(pos):
            d = np.dot(p - cm, n)
            d_list.append((i, d))

        # Sort the atoms by their signed distance
        d_list.sort(key=lambda x: x[1])
        nu_taken = dict()

        # Select half of the atoms needed for a full cluster
        p_use = []
        n_use = []
        for (i, d) in d_list:
            if num[i] not in nu_taken.keys():
                nu_taken[num[i]] = 0
            if nu_taken[num[i]] < nu[num[i]] / 2.:
                p_use.append(pos[i])
                n_use.append(num[i])
                nu_taken[num[i]] += 1

        # calculate the mirrored position and add these.
        pn = []
        for p in p_use:
            pt = p - 2. * np.dot(p - cm, n) * n
            if reflect:
                pt = -pt + 2 * cm + 2 * n * np.dot(pt - cm, n)
            pn.append(pt)

        n_use.extend(n_use)
        p_use.extend(pn)

        # In the case of an uneven number of
        # atoms we need to add one extra
        for n in nu.keys():
            if nu[n] % 2 == 0:
                continue
            while sum(n_use == n) > nu[n]:
                for i in range(int(len(n_use) / 2), len(n_use)):
                    if n_use[i] == n:
                        del p_use[i]
                        del n_use[i]
                        break
            assert sum(n_use == n) == nu[n]

        # Make sure we have the correct number of atoms
        # and rearrange the atoms so they are in the right order
        for i in range(len(n_use)):
            if num[i] == n_use[i]:
                continue
            for j in range(i + 1, len(n_use)):
                if n_use[j] == num[i]:
                    tn = n_use[i]
                    tp = p_use[i]
                    n_use[i] = n_use[j]
                    p_use[i] = p_use[j]
                    p_use[j] = tp
                    n_use[j] = tn

        # Finally we check that nothing is too close in the end product.
        cand = Atoms(num, p_use, cell=slab.get_cell(), pbc=slab.get_pbc())
        tc = atoms_too_close(cand, blmin)
        if tc:
            continue
        tc = atoms_too_close_two_sets(slab, cand, blmin)
        if not changed and counter > n_tries // 2:
            reflect = not reflect
            changed = True
        tot = cand + slab
    if counter == n_tries:
        return None, mirror.__name__

    return tot, mirror.__name__


def rotate(atoms, blmin, n_top=None, maxcount=mc, fraction=0.33, min_angle=1.57, test_dist_to_slab=True, rng=np.random):
    """Does the actual mutation."""
    n = len(atoms) if n_top is None else n_top
    slab = atoms[:len(atoms) - n]
    atoms = atoms[-n:]

    mutant = atoms.copy()
    pos = mutant.get_positions()
    tags = mutant.get_tags()
    eligible_tags = tags if tags is None else tags

    indices = {}
    for tag in np.unique(tags):
        hits = np.where(tags == tag)[0]
        if len(hits) > 1 and tag in eligible_tags:
            indices[tag] = hits

    n_rot = int(np.ceil(len(indices) * fraction))
    chosen_tags = rng.choice(list(indices.keys()), size=n_rot,
                                  replace=False)

    too_close = True
    count = 0
    while too_close and count < maxcount:
        newpos = np.copy(pos)
        for tag in chosen_tags:
            p = np.copy(newpos[indices[tag]])
            cop = np.mean(p, axis=0)

            if len(p) == 2:
                line = (p[1] - p[0]) / np.linalg.norm(p[1] - p[0])
                while True:
                    axis = rng.rand(3)
                    axis /= np.linalg.norm(axis)
                    a = np.arccos(np.dot(axis, line))
                    if np.pi / 4 < a < np.pi * 3 / 4:
                        break
            else:
                axis = rng.rand(3)
                axis /= np.linalg.norm(axis)

            angle = min_angle
            angle += 2 * (np.pi - min_angle) * rng.rand()

            m = get_rotation_matrix(axis, angle)
            newpos[indices[tag]] = np.dot(m, (p - cop).T).T + cop

        mutant.set_positions(newpos)
        mutant.wrap()
        too_close = atoms_too_close(mutant, blmin)
        count += 1

        if not too_close and test_dist_to_slab:
            too_close = atoms_too_close_two_sets(slab, mutant, blmin)

    if count == maxcount:
        return None, rotate.__name__
    else:
        mutant = slab + mutant

    return mutant, rotate.__name__



