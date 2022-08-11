import os
import math

import torch
import pickle
import sklearn
import gpytorch
import ase.io
import ase.io.vasp
import numpy as np
import pandas as pd
from glob import glob
from scipy import stats
from dscribe.descriptors import MBTR, ValleOganov
import multiprocess
from sklearn.model_selection import train_test_split
from sklearn.metrics import (mean_absolute_error,
                             mean_squared_error)


class MLMemory:
    def __init__(self, descriptor=None, e_pt=-379.70, conf=None):
        self.properties = ['structure', 'features', 'ene', 'for_ene', 'gen']

        self.descriptor = descriptor
        self.datasheet = None

        self.ene_trans = ForEneTransfer(e_pt)

        self.conf = conf

    def save_data(self, atoms_list, energy_list, gen, save_loc=None):
        self.update_all(atoms_list, energy_list, gen)

        if save_loc:
            f = open(save_loc, 'wb')
            pickle.dump(self.datasheet, f, protocol=3)
            f.close()

    def update_all(self, atoms_list, energy_list, gen):
        assert len(atoms_list) == len(energy_list)
        feature_mat = self.descriptor.create(atoms_list, n_jobs=int(multiprocess.cpu_count()))  # Parallel
        temp = atoms_list[0]
        n_O = np.count_nonzero(temp.get_atomic_numbers() == 8)

        for_ene_list = self.ene_trans.to_for_ene(atoms_list, energy_list, n_O)
        if type(gen) is not list:
            gen_list = [gen] * len(atoms_list)
        else:
            gen_list = gen

        if self.datasheet is None:
            self.datasheet = pd.DataFrame(list(zip(atoms_list, feature_mat, energy_list, for_ene_list, gen_list)),
                                          columns=self.properties)
        else:
            df_plus = pd.DataFrame(list(zip(atoms_list, feature_mat, energy_list, for_ene_list, gen_list)),
                                   columns=self.properties)

            self.datasheet = pd.concat([self.datasheet, df_plus], ignore_index=True)

        self.datasheet = dropduplicates(self.datasheet, 'structure')

        df_sort = self.datasheet.sort_values(by='ene', ascending=False)
        self.datasheet = df_sort[int(len(df_sort) * (1 - self.conf.ene_shield)):]

        if len(self.datasheet) > self.conf.max_data_size:
            df_sort = self.datasheet.sort_values(by='gen')
            self.datasheet = df_sort[-self.conf.max_data_size:]

    def load_dataset(self, df_file):
        df = open(df_file, 'rb')

        if self.datasheet is None:
            self.datasheet = pickle.load(df)
        else:
            df_plus = pickle.load(df)
            self.datasheet = pd.concat([self.datasheet, df_plus], ignore_index=True)

    def dataset_nan_reload(self, conf):
        atoms_list = list()
        energy_list = list()
        gen = list()

        for now_clu in range(0, conf.num_cluster):
            path_clu = os.path.join(conf.home_path, 'Cluster' + str(now_clu))
            gen_path = os.path.join(path_clu, "Gen0")

            file = open(os.path.join(gen_path, "GAlog.txt"), 'r')
            galog = [[int(x.split()[0]), int(x.split()[1]), float(x.split()[2])] for x in file]
            atoms_list = atoms_list + [ase.io.read(os.path.join(gen_path, str(int(x[1])), "CONTCAR")) for x in galog]
            energy_list = energy_list + [float(x[2]) for x in galog]

        gen = gen + [0] * len(energy_list)

        fs_1 = glob(os.path.join(conf.home_path, '*', '*', '*', 'OUTCAR'))

        stru_list = list()

        for car in fs_1:
            try:
                stru_list.append(ase.io.vasp.read_vasp_out(car))
                gen.append(int(car.split('/')[-3][3:]))
            except:
                continue

        atoms_list = atoms_list + stru_list
        energy_list = energy_list + [x.get_potential_energy() for x in stru_list]

        self.save_data(atoms_list, energy_list, gen,
                       save_loc=os.path.join(conf.home_path, conf.data_save_file))

class GPR:
    def __init__(self, conf, log):
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.descriptor = ValleOganov(
            species=["Pt", "O"],
            k2={
                "sigma": 10**(-0.5),
                "n": 100,
                "r_cut": 5
            },
            k3={
                "sigma": 10**(-0.5),
                "n": 100,
                "r_cut": 5
            },
        )

        self.model = None
        self.optimizer = None

        self.conf = conf
        self.log = log
        self.memory = MLMemory(self.descriptor, e_pt=self.conf.e_pt, conf=self.conf)
        self.training_iter = self.conf.training_iter

        self.bias = None

    def sampling(self, strucutre_list, n_samples=1):
        temp = strucutre_list[0]
        n_O = np.count_nonzero(temp.get_atomic_numbers() == 8)

        x = self.descriptor.create(strucutre_list, n_jobs=int(multiprocess.cpu_count()))
        x = torch.tensor(x, dtype=torch.float32)

        self.model.eval()
        self.likelihood.eval()

        with torch.no_grad():
            observed_pred = self.likelihood(self.model(x))
            e = observed_pred.mean
            std = observed_pred.stddev

        # acquisition_func = self.model(x).sample(torch.Size([n_samples]))
        px = torch.distributions.Normal(e, std)
        acquisition_func = px.sample()

        e = self.memory.ene_trans.to_ene(e, n_O)
        acquisition_func = self.memory.ene_trans.to_ene(acquisition_func, n_O)

        return e, std, acquisition_func

    def train(self, request_train=True):
        if self.memory.datasheet is None:
            self.memory.load_dataset(os.path.join(self.conf.home_path, self.conf.data_save_file))
            if self.memory.datasheet is None:
                self.log.log_msg += f"Dataset Load Failed, restart may re-construct dataset \n"
                self.memory.dataset_nan_reload(self.conf)

        sheet_train, sheet_test = train_test_split(self.memory.datasheet, test_size=0.1)

        features_x_train = torch.tensor(np.array(sheet_train['features'].tolist()), dtype=torch.float32)
        y_train = torch.tensor(np.array(sheet_train['for_ene'].tolist()), dtype=torch.float32)

        self.model = ExactGPModel(features_x_train, y_train, self.likelihood)

        # if os.path.exists(os.path.join(self.conf.home_path, self.conf.model_save_file)):
        #     state_dict = torch.load(os.path.join(self.conf.home_path, self.conf.model_save_file), map_location='cpu')
        #     self.model.load_state_dict(state_dict)
        #
        #     for param_name, param in self.model.named_parameters():
        #         print(f'Parameter name: {param_name:42} value = {param.item()}')
        #
        #     self.log.log_msg += f'Read ML model parameter from ' \
        #                         f'{os.path.join(self.conf.home_path, self.conf.model_save_file)} \n'

        self.optimizer = torch.optim.Adam(self.model.parameters(), amsgrad=True, lr=0.004)

        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        self.log.log_msg += f"Dataset of training point: {len(self.memory.datasheet)} \n"

        if request_train:
            for iter in range(self.training_iter):
                try:
                    self.optimizer.zero_grad()
                    output = self.model(features_x_train)
                    loss = -mll(output, y_train)
                    loss.backward()
                    if iter == 0:
                        iter_0 = loss.item()
                    elif iter == self.training_iter-1:
                        iter_end = loss.item()
                    self.optimizer.step()
                except:
                    self.log.log_msg += f'ML training failed, load parameter from ' \
                                        f'{os.path.join(self.conf.home_path, self.conf.model_save_file)} \n'
                    state_dict = torch.load(os.path.join(self.conf.home_path, self.conf.model_save_file))
                    self.model.load_state_dict(state_dict)

            self.log.log_msg += f'GPR loss change from {iter_0:.2f} to {iter_end:.2f} \n'

        # features_X_test
        features_X_test = torch.tensor(np.array(sheet_test['features'].tolist()), dtype=torch.float32)
        y_test = torch.tensor(np.array(sheet_test['ene'].tolist()), dtype=torch.float32)

        # Get into evaluation (predictive posterior) mode
        self.model.eval()
        self.likelihood.eval()

        f_preds = self.likelihood(self.model(features_X_test))
        targets_pred = f_preds.mean.detach().numpy()
        targets_pred = self.memory.ene_trans.to_ene(targets_pred)

        targets_val = y_test.cpu().detach().numpy()
        mae = mean_absolute_error(targets_val, targets_pred)
        rmse = np.sqrt(mean_squared_error(targets_val, targets_pred))

        self.log.log_msg += f'MAE on val set {mae:2f} and RMSE {rmse:.2f} \n'

        torch.save(self.model.state_dict(), os.path.join(self.conf.home_path, self.conf.model_save_file))

        for param_name, param in self.model.named_parameters():
            print(f'Parameter name: {param_name:42} value = {param.item()}')

# We will use the simplest form of GP model, exact inference
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel() + gpytorch.kernels.LinearKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class ForEneTransfer:
    def __init__(self, e_pt):
        self.e_h2 = -6.7593679
        self.e_pt = e_pt
        self.e_h2O = -14.2
        self.n_O = None

    def to_for_ene(self, atoms_list, ene_list, *args):
        if args:
            self.n_O = args[0]
        for_ene_list = list()
        for stru, x in zip(atoms_list, ene_list):
            for_ene = ((x + self.n_O * self.e_h2) - (self.e_pt + self.n_O * self.e_h2O)) / self.n_O
            for_ene_list.append(for_ene)
        return for_ene_list

    def to_ene(self, ene_list, *args):
        if args:
            self.n_O = args[0]
        return ene_list * self.n_O + (self.e_pt + self.n_O * self.e_h2O) - (self.n_O * self.e_h2)


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
    np.savez(os.path.join(home_path, 'gpr_list.npz'),
             rmse=np.array(rmse_list), tau=np.array(tau_list), std=np.array(std_list))

def dropduplicates(dataframe: pd.DataFrame,
                     columnOfStructure: str,
                     eletype: str = 'ase') -> pd.DataFrame:
    dataframe['dropcol'] = [
        (len(struc), struc.arrays['positions'].tobytes(),
         struc.arrays['numbers'].tobytes(), struc.cell.tobytes(),
         struc.pbc.tobytes())
        if eletype == 'ase' else Structure.to(struc, fmt='cif')
        for struc in dataframe[columnOfStructure].values
    ]
    dataframe.drop_duplicates(subset=['dropcol'], inplace=True)
    dataframe.drop(columns=['dropcol'], inplace=True)
    return dataframe
