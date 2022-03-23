# -*- coding: utf-8 -*-
import os
from time import time

from candidates.candidates_surface import Initializer, FollowingGen
from log import LogGenerator
from sampling import GPRSampling

class HeterogaSurface:
    def __init__(self, conf):
        self.conf = conf
        self.log = LogGenerator(conf=self.conf)

        if self.conf.sampling == 'GPR':
            self.samp = GPRSampling(self.log, self.conf)

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

    def run(self):
        self.conf.gen = self.check_gen_number()  # Restart
        self.log.genrate_basic_info(self.conf.gen)

        if self.conf.gen == 0:
            self.initializer()
            self.conf.gen += 1

        self.samp.initial()

        while self.conf.gen < self.conf.num_gen:
            t0 = time()
            self.log.log_msg += f"Now we will start to calculate {str(self.conf.gen)} generation \n"
            self.following_gen()
            t1 = time()
            m, s = divmod(t1 - t0, 60)
            self.log.log_msg += f"Time spend for this generation: {m:.0f}min {s:.0f}s \n"
            self.log.log_msg += f'Calculation at Gen {self.conf.gen} Finished \n\n'
            self.log.save_log()
            self.conf.gen += 1
