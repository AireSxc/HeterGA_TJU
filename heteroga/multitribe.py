import math
import os
import random
import shutil

from log import cal_fitness


def exchange(gen, now_cluster, conf, log_set, mode='roulette_choose'):

    path_cluster = os.path.join(conf.home_path, 'Cluster' + str(now_cluster))
    path_gen = os.path.join(path_cluster, "Gen" + str(gen))

    if mode == 'roulette_choose': # TODO: choose_best
        random_list = list(range(conf.num_cluster))
        random_list.remove(now_cluster)
        target_cluster = random.choice(random_list)
        list_for_comm = log_set.gen_loclist_with_cluster[target_cluster]
        fit_comm = cal_fitness(list_for_comm, conf.num_fit)

        p0 = None
        while p0 is None:
            p_t = random.randrange(0, int(math.ceil(conf.num_fit * 0.3))) if gen == 1 else random.randrange(1, conf.num_fit)
            if random.randrange(0, 10000) / 10000.0 < fit_comm[p_t]:
                p0 = p_t

    target_path = os.path.join(conf.home_path, 'Cluster' + str(target_cluster),
                               "Gen" + str(int(list_for_comm[p0][0])), str(int(list_for_comm[p0][1])))
    target_energy = list_for_comm[p0][2]

    terminal_path = os.path.join(path_gen, '0')
    if not os.path.exists(terminal_path):
        os.makedirs(terminal_path)

    f_list = os.listdir(target_path)

    target_file = None
    for item in f_list:
        if os.path.splitext(item)[1] == '.cif':
            target_file = item

    shutil.copy(os.path.join(target_path, target_file), terminal_path)
    shutil.copy(os.path.join(target_path, 'CONTCAR'), terminal_path)

    os.chdir(terminal_path)
    cmd = "echo " + str(gen) + "  0 " + str(target_energy) + " >> ../GAlog.txt"
    os.system(cmd)