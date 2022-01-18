# -*- coding: utf-8 -*-
import os
from abc import ABC, abstractmethod

from candidates.candidates_surface import Initializer, FollowingGen
from log import LogGenerator
from sampling import GPRSampling


class HeterogaOperation(ABC):
    def __init__(self):
        self.gen = 0

    def check_gen_number(self, home_path):
        gen = 0
        while os.path.exists(home_path + f'/Cluster0/Gen{gen}'):
            gen += 1
        return gen

    def model_train(self):
        """ Method to be implemented for the operations that rely on
        a Machine-Learned model to perform more informed/guided
        mutation and crossover operations.
        """
        pass


class HeterogaSurface(HeterogaOperation):
    def __init__(self, conf):
        HeterogaOperation.__init__(self)

        self.conf = conf
        self.log = LogGenerator(conf=self.conf)

        if self.conf.sampling == 'GPR':
            self.samp = GPRSampling(self.log, self.conf)

    def initializer(self):
        ini = Initializer(conf=self.conf, calc=self.conf.calc, log_set=self.log)
        ini.run()

    def following_gen(self):
        fol = FollowingGen(gen=self.gen, conf=self.conf, log_set=self.log)
        fol.run()

    def run(self):
        self.gen = self.check_gen_number(self.conf.home_path)  # Restart

        if self.gen == 0:
            self.initializer()

        self.samp.initial()

        while self.gen < self.conf.num_gen:
            self.following_gen()
            self.log.log_msg += f"Now we will start to calculate {str(self.gen)} th generation \n"
            self.gen += 1
