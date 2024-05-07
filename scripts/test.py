from ControllableNesymres.utils import load_metadata_hdf5, retrofit_word2id, return_fitfunc
from ControllableNesymres.architectures.data import ControllableNesymresDataset
import ControllableNesymres.architectures.data as Data

import torch
import omegaconf
import pandas as pd
import sympy as sp

import numpy as np
import json

#TO DELETE
import math
import random

from pathlib import Path
from functools import partial
import argparse


        
def main(model_type, min_support, max_support, test_path):
    seed = 22

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    operators = {
    'abs': np.abs,
    'acos': np.arccos,
    'add': np.add,
    'asin': np.arcsin,
    'atan': np.arctan,
    'cos': np.cos,
    'cosh': np.cosh,
    'coth': lambda x: 1/np.tanh(x),
    'div': np.divide,
    'exp': np.exp,
    'inv': np.reciprocal,
    'ln': np.log,
    'mul': np.multiply,
    'pow': np.power,
    'pow2': lambda x: np.power(x, 2),
    'pow3': lambda x: np.power(x, 3),
    'pow4': lambda x: np.power(x, 4),
    'pow5': lambda x: np.power(x, 5),
    'sin': np.sin,
    'sinh': np.sinh,
    'sqrt': np.sqrt,
    'sub': np.subtract,
    'tan': np.tan,
    'tanh': np.tanh
    }

    if model_type == "mmsr":
        cfg =  omegaconf.OmegaConf.load(Path("configs/mmsr_config.yaml"))
        hardware_cfg = omegaconf.OmegaConf.load('configs/host_system_config/host.yaml')
        cfg = omegaconf.OmegaConf.merge(cfg, hardware_cfg)

        model = 'weights/10000000_log_-epoch=201-val_loss=0.00_new_version.ckpt'
    else:
        cfg = omegaconf.OmegaConf.load(Path('configs/nsr_network_config.yaml'))
        model = 'model/ControllableNeuralSymbolicRegressionWeights/nsr_200000000_epoch=149.ckpt'

    cfg.inference.bfgs.activated = False
    cfg.inference.bfgs.n_restarts=10
    cfg.inference.n_jobs=1
    cfg.dataset.fun_support.max = max_support
    cfg.dataset.fun_support.min = min_support
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

    evaluations_list = []

    x_1, x_2, x_3, x_4, x_5 = sp.symbols('x_1 x_2 x_3 x_4 x_5')

    for idx, inputs in enumerate(testloader):
        if idx == 0 or inputs is None:
            continue
        b = inputs[0].permute(0, 2, 1).to("cuda")
        X = b[:, :, :-1]
        y = b[:, :, -1]    

        X = X.half()
        outputs = fitfunc(X, y, cond, is_batch=True)

        assert y[0].shape[0] == X.squeeze().shape[0], "The number of samples in the input and output tensors are not equal"
        evaluation_dict = {
            'Number of Samples': y[0].shape[0],
            'True Equation': inputs[2][0][0],
            'Predicted Equation': outputs['best_pred']
        }

        if outputs['best_pred'] == 'illegal parsing infix':
            evaluation_dict.update({
            'Match Equation': 0,
            'R2 Score': float('nan'),
            'Error Message': 'Equation is syntactically incorrect'
        })
            evaluations_list.append(evaluation_dict)
            continue

        variables = X[0, :, :].cpu()
        equation = sp.lambdify((x_1, x_2, x_3, x_4, x_5), outputs['best_pred'], modules=['numpy', operators])

        try:
            results = equation(*variables.permute(1, 0))
        except NameError as e:
            evaluation_dict.update({
            'Match Equation': 0,
            'R2 Score': float('nan'),
            'Error Message': f'NameError: {e}'
        })
        else:
            if isinstance(results, (int, float)) and results == 0:
                results = torch.zeros(variables.shape[0])

            r2 = r2_score(y[0].cpu(), results)

            evaluation_dict.update({
            'Match Equation': 1 if r2 > 0.99 else 0,
            'R2 Score': r2.item(),
            'Error Message': ''
        })

        print('True equation: ', inputs[2][0][0])
        print('Predicted equation: ', outputs['best_pred'])
        evaluations_list.append(evaluation_dict)


    evaluation_of_prediction = pd.DataFrame(evaluations_list)
    
    path = 'evaluation/'
    file_name = f'evaluation_of_prediction_{model_type}_{Path(test_path).name}_SupportRange{cfg.dataset.fun_support.min}to{cfg.dataset.fun_support.max}'

    filename_csv = f'{file_name}.csv'

    config = {
        'minimum_support': cfg.dataset.fun_support.min,
        'maximum_support': cfg.dataset.fun_support.max,
        'model': model_type,
        'dataset': Path(test_path).name,
        'csv_filename': filename_csv
    }

    filename_config = f'{path}{file_name}.json'

    with open(filename_config, 'w') as f:
        json.dump(config, f)

    evaluation_of_prediction.to_csv(f'{path}{filename_csv}', index=False)

def r2_score(y_true, y_pred):
    assert torch.isnan(y_true).any().item() == False, "y_true has NaN values"
    if torch.isnan(y_pred).any().item():
        return torch.tensor([0])

    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process the arguments model type, support range, data path')

    parser.add_argument('--min_support', type=int, default=-10, help='The minimum support value')
    parser.add_argument('--max_support', type=int, default=10, help='The maximum support value')
    parser.add_argument('--model_type', type=str, default='mmsr', help='The model type')
    parser.add_argument('--test_path', type=str, default='data/benchmark/train_nc', help='The path of the test data')

    args = parser.parse_args()

    main(args.model_type, args.min_support, args.max_support, args.test_path)