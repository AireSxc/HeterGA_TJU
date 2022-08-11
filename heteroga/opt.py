# -*- coding: utf-8 -*-
import os
import sys
import time
import copy
from random import sample

from candidates.candidates_surface import Initializer, FollowingGen
from log import LogGenerator
from sampling import GPRSampling
from utilities import EarlyStopping

class HeterogaSurface:
    def __init__(self, conf):
        self.conf = conf
        self.log = LogGenerator(conf=self.conf)

        if self.conf.sampling == 'GPR':
            self.samp = GPRSampling(self.log, self.conf)

        self.early_stopping = EarlyStopping()

    def check_gen_number(self):
        cluster_gen = list()
        for now_cluster in range(self.conf.num_cluster):
            gen = 0
            while os.path.exists(os.path.join(self.conf.home_path, 'Cluster' + str(now_cluster), 'Gen' + str(gen))):
                gen += 1
            cluster_gen.append(gen)

        return min(cluster_gen)

    def initializer(self):
        ini = Initializer(conf=self.conf, log_set=self.log)
        ini.run()

    def following_gen(self):
        fol = FollowingGen(gen=self.conf.gen, conf=self.conf, log_set=self.log, samp=self.samp)
        fol.run()
        self.early_stop(self.log.lowest_energy)

    def early_stop(self, target_val):
        self.early_stopping(target_val)
        if self.early_stopping.early_catastrophe:
            self.log.log_msg += f"Trigger Catastrophe \n"
            self.catastrophe(self.conf.gen)
        elif self.early_stopping.early_end:
            self.log.log_msg += f"Trigger End \n"
            self.log.save_log()
            sys.exit()

    def catastrophe(self, gen, top_fix_num=5, sample_num=8):
        for clu in range(0, self.conf.num_cluster):
            new_file = open(os.path.join(self.conf.home_path, f"Cluster{str(clu)}", f'rank_after_G{str(gen)}.txt'))
            new_file_list = [[int(x.split()[0]), int(x.split()[1]), float(x.split()[2])] for x in new_file]
            new_file.close()

            ignore_num = [i[1] for i in new_file_list if i[0] == 0]

            old_file = open(os.path.join(self.conf.home_path, f"Cluster{str(clu)}", f'rank_after_G0.txt'))
            old_file_list = [[int(x.split()[0]), int(x.split()[1]), float(x.split()[2])] for x in old_file]
            old_file_list = [old_file_list[i] for i in range(len(old_file_list)) if old_file_list[i][1] not in ignore_num]
            old_file.close()

            ran = sample(range(len(new_file_list[top_fix_num:])), sample_num)
            ran = [i + top_fix_num for i in ran]

            diff = new_file_list[0][2] - old_file_list[0][2]

            cata_file_list = list()
            count = 0

            for i in range(len(new_file_list)):
                if i in ran and new_file_list[i][0] != gen:
                    count += 1
                    n = copy.deepcopy(old_file_list[count])
                    n[2] = n[2] + diff
                    cata_file_list.append(n)
                else:
                    cata_file_list.append(new_file_list[i])

            os.rename(os.path.join(self.conf.home_path, f"Cluster{str(clu)}", f'rank_after_G{str(gen)}.txt'),
                      os.path.join(self.conf.home_path, f"Cluster{str(clu)}", f'rank_after_G{str(gen)}_before_catastrophe.txt'))

            fit_file = open(os.path.join(self.conf.home_path, f"Cluster{str(clu)}", f'rank_after_G{str(gen)}.txt'), "w")

            for n in range(len(cata_file_list)):
                fit_file.write("%4d%4d%15.7f\n" % (cata_file_list[n][0], cata_file_list[n][1], cata_file_list[n][2]))
            fit_file.close()

        self.early_stopping.early_catastrophe = False

    def run(self):
        self.conf.gen = self.check_gen_number()  # Restart
        self.log.genrate_basic_info(self.conf.gen)

        if self.conf.gen == 0:
            self.initializer()
            self.conf.gen += 1
        else:
            self.log.restart_log(self.conf.gen)

        self.samp.initial()
        self.log.save_log()

        while self.conf.gen < self.conf.num_gen:
            t0 = time.time()
            self.log.log_msg += f"Now we will start to calculate {str(self.conf.gen)} generation \n"
            self.following_gen()
            t1 = time.time()
            m, s = divmod(t1 - t0, 60)
            self.log.log_msg += f"Time spend for this generation: {m:.0f}min {s:.0f}s \n"
            self.log.log_msg += f'Calculation at Gen {self.conf.gen} Finished. ' \
                                f'Time: {time.asctime(time.localtime(time.time()))} \n\n'
            self.log.save_log()
            self.conf.gen += 1
