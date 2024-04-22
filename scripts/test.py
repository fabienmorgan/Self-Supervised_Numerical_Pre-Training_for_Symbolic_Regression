from ControllableNesymres.utils import load_metadata_hdf5, retrofit_word2id, return_fitfunc
from ControllableNesymres.architectures.data import ControllableNesymresDataset
import ControllableNesymres.architectures.data as Data

import torch
import omegaconf
import pandas as pd
import os
import sympy as sp

import random
import numpy as np
import math

from pathlib import Path
from functools import partial


def robust_dataloader(test_loader):
    iter_test_loader = iter(test_loader)
    while True:
        try:
            yield next(iter_test_loader)
        except StopIteration:
            return 
        except Exception as e:
            print(e)
            pass 
        
def main():
    """seed = 22 #6, 11

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)"""

    model_type = "mmsr"
    test_path = 'data/benchmark/train_nc'

    if model_type == "mmsr":
        cfg =  omegaconf.OmegaConf.load(Path("configs/mmsr_config.yaml"))
        hardware_cfg = omegaconf.OmegaConf.load('configs/host_system_config/host.yaml')
        cfg = omegaconf.OmegaConf.merge(cfg, hardware_cfg)

        model = 'weights/10000000_log_-epoch=94-val_loss=0.00.ckpt'
    else:
        cfg = omegaconf.OmegaConf.load(Path('configs/nsr_network_config.yaml'))
        model = 'model/ControllableNeuralSymbolicRegressionWeights/nsr_200000000_epoch=149.ckpt'

    cfg.inference.bfgs.activated = False
    cfg.inference.bfgs.n_restarts=10
    cfg.inference.n_jobs=1
    cfg.dataset.fun_support.max =5
    cfg.dataset.fun_support.min = -5
    cfg.inference.beam_size = 5

    metadata = load_metadata_hdf5(Path(test_path))
    metadata = retrofit_word2id(metadata, cfg)

    data = ControllableNesymresDataset(Path(test_path), cfg, 'test')

    testloader = torch.utils.data.DataLoader(
            data,
            batch_size=1,
            shuffle=False,
            collate_fn=partial(Data.custom_collate_fn,total_variables=data.total_variables, total_coefficients=data.total_coefficients,cfg=cfg),
            num_workers=cfg.num_of_workers,
            pin_memory=True,
            drop_last=False
        )

    if torch.cuda.is_available():
        fitfunc = return_fitfunc(cfg, metadata, model, device="cuda")
    else:
        fitfunc = return_fitfunc(cfg, metadata, model, device="cpu")

    cond = {'symbolic_conditioning': torch.tensor([[1, 2]]), 'numerical_conditioning': torch.tensor([])}
    cond["symbolic_conditioning"] = cond["symbolic_conditioning"].unsqueeze(0)  
    cond["numerical_conditioning"] = cond["numerical_conditioning"].unsqueeze(0) 

    true_equations = []
    predicted_equations = []
    match_equations = []
    r2_scores = []

    count = 0

    x_1, x_2, x_3, x_4, x_5 = sp.symbols('x_1 x_2 x_3 x_4 x_5')

    for idx, inputs in enumerate(robust_dataloader(testloader)):
        try:
            if idx == 0:
                continue
            b = inputs[0].permute(0, 2, 1).to("cuda")
            X = b[:, :, :-1]
            y = b[:, :, -1]    

            X = X.half()
            outputs = fitfunc(X, y, cond, is_batch=True)

            variables = X[0, :, :].cpu()

            x_1, x_2, x_3, x_4, x_5 = sp.symbols('x_1 x_2 x_3 x_4 x_5')

            equation = sp.lambdify((x_1, x_2, x_3, x_4, x_5), outputs['best_pred'], 'numpy')

            results = equation(variables[:,0], variables[:,1], variables[:,2], variables[:,3], variables[:,4])

            r2 = r2_score(y[0].cpu(), results)

            if r2 > 0.99:
                match_equations.append(1)
            else:
                match_equations.append(0)

            print('True equation: ', inputs[2][0][0])
            print('Predicted equation: ', outputs['best_pred'])
            true_equations.append(inputs[2][0][0])
            predicted_equations.append(outputs['best_pred'])
            r2_scores.append(r2.item())

        except Exception as e:
            print(e)
            pass

    evaluation_of_prediction = pd.DataFrame({
    'True equation': true_equations,
    'Predicted equation': predicted_equations,
    'Match Equation': match_equations,
    'R2 Score': r2_scores
    })
    
    evaluation_of_prediction.to_csv(f'evaluation_of_prediction_{model_type}.csv', index=False)

def r2_score(y_true, y_pred):
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot


if __name__ == '__main__':
    main()