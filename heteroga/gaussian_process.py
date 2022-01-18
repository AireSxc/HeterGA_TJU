import os
import math
import pyximport
pyximport.install()

import ase.io
import sklearn
import numpy as np
from scipy import stats
from dscribe.descriptors import SOAP
from sklearn.gaussian_process.kernels import RBF, Matern
from sklearn.gaussian_process import GaussianProcessRegressor

from prior.prior import RepulsivePrior

class MLMemory:
    def __init__(self, descriptor=None, prior=None):

        self.descriptor = descriptor

        if prior is None:
            self.prior = RepulsivePrior()
        else:
            self.prior = prior

        self.energies = None
        self.features = None
        self.prior_values = None

    def save_data(self, atoms_list, energy_list, save_loc=None):
        self.update_feature(atoms_list)
        self.update_prior(atoms_list)
        self.update_energy(energy_list)

        if save_loc is not None:
            np.savez(os.path.join(save_loc, 'GPR_para.npz'),
                     features=self.features, energies=self.energies, prior_values=self.prior_values)

    def update_feature(self, atoms_list):
        print(len(atoms_list))
        feature_mat = self.descriptor.create(atoms_list, n_jobs=len(atoms_list))  # Parallel
        if self.features is None:
            self.features = np.array(feature_mat)
        else:
            self.features = np.r_[self.features, np.array(feature_mat)]

    def update_prior(self, atoms_list):
        prior_values_save = np.array([self.prior.energy(a) for a in atoms_list])
        if self.prior_values is None:
            self.prior_values = prior_values_save
        else:
            self.prior_values = np.r_[self.prior_values, prior_values_save]

    def update_energy(self, energy_list):
        if self.energies is None:
            self.energies = np.array(energy_list)
        else:
            self.energies = np.r_[self.energies, np.array(energy_list)]

    def load_para(self, npz_file):
        para = np.load(npz_file)
        self.features = para['features']
        self.energies = para['energies']
        self.prior_values = para['prior_values']


class GPR:
    def __init__(self, sample_stru=None, descriptor=None, model=None, prior=None):

        if model is None:
            kernel = Matern(length_scale=10.0, length_scale_bounds=(1e0, 1e3))
            self.model = GaussianProcessRegressor(kernel=kernel)
        else:
            self.model = model

        if descriptor is None:
            sample_stru = ase.io.read(sample_stru)
            species = list(set(sample_stru.get_chemical_symbols()))
            self.descriptor = SOAP(species=species, periodic=True, rcut=4.0, nmax=5, lmax=5, average='inner')
        else:
            self.descriptor = descriptor

        if prior is None:
            self.prior = RepulsivePrior()
        else:
            self.prior = prior

        self.memory = MLMemory(self.descriptor, self.prior)
        self.bias = None

    def predict_energy(self, strucutre_list, eval_std=False):
        """Evaluate the energy predicted by the GPR-model.

        parameters:

        a: Atoms object
            The structure to evaluate.

        eval_std: bool
            In addition to the force, predict also force contribution
            arising from including the standard deviation of the
            predicted energy.
        """
        x = self.descriptor.create(strucutre_list, n_jobs=len(strucutre_list))
        y_pred, sigma = self.model.predict(x, return_std=True)
        e = y_pred + self.bias + np.array([self.prior.energy(i) for i in strucutre_list])

        if eval_std:
            return e, sigma
        else:
            return e

    def thompson_sampling(self, strucutre_list, n_samples=1):
        x = self.descriptor.create(strucutre_list, n_jobs=len(strucutre_list))
        y_pred, cov = self.model.predict(x, return_cov=True)

        # Sampling
        rng = sklearn.utils.check_random_state(0)
        y_samples = rng.multivariate_normal(y_pred, cov, n_samples).T
        posterior_sample = y_samples.T[0]
        prior_list = np.array([self.prior.energy(i) for i in strucutre_list])
        acquisition_func = posterior_sample + prior_list + self.bias
        e = y_pred + self.bias + prior_list

        return e, acquisition_func

    def train(self):
        self.bias = np.mean(self.memory.energies - self.memory.prior_values)
        y = self.memory.energies - self.memory.prior_values - self.bias

        self.model.fit(self.memory.features, y)


def load_gpr_recode(gpr_recode_file):
    para = np.load(gpr_recode_file)
    return para['rmse'].tolist(), para['tau'].tolist(), para['std'].tolist()


def gpr_recode(home_path, error_all_list, rmse_list, tau_list, std_list):
    pred_list = error_all_list[..., 0]
    calc_list = error_all_list[..., 1]
    rmse = math.sqrt(sklearn.metrics.mean_squared_error(pred_list, calc_list))
    print('RMSE:' + str(rmse))

    tau, _ = stats.kendalltau(pred_list, calc_list)

    rmse_list.append(rmse)
    tau_list.append(tau)
    std_list.append(np.mean(error_all_list[..., 2]))
    np.savez(home_path + '/gpr_list.npz', rmse=np.array(rmse_list), tau=np.array(tau_list), std=np.array(std_list))


