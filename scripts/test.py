from ControllableNesymres.utils import load_metadata_hdf5, retrofit_word2id, return_fitfunc
from ControllableNesymres.architectures.data import ControllableNesymresDataset
import ControllableNesymres.architectures.data as Data

import torch
import omegaconf
from pathlib import Path
from functools import partial

def main():
    model_type = "nsr"

    if model_type == "mmsr":
        cfg =  omegaconf.OmegaConf.load(Path("configs/mmsr_config.yaml"))
        hardware_cfg = omegaconf.OmegaConf.load('configs/host_system_config/host.yaml')
        cfg = omegaconf.OmegaConf.merge(cfg, hardware_cfg)

        model = 'weights/200000_log_-epoch=11-val_loss=0.00.ckpt'
    else:
        cfg = omegaconf.OmegaConf.load(Path('configs/nsr_network_config.yaml'))
        model = 'model/ControllableNeuralSymbolicRegressionWeights/nsr_200000000_epoch=149.ckpt'

    cfg.inference.bfgs.activated = True
    cfg.inference.bfgs.n_restarts=10
    cfg.inference.n_jobs=-1
    cfg.dataset.fun_support.max =5
    cfg.dataset.fun_support.min = -5
    cfg.inference.beam_size = 5

    metadata = load_metadata_hdf5(Path('data/benchmark/train_nc'))
    metadata = retrofit_word2id(metadata, cfg)

    data = ControllableNesymresDataset(Path('data/benchmark/train_nc'), cfg, 'test')

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

    for inputs in testloader:
        b = inputs[0].permute(0, 2, 1).to("cuda")
        size = b.shape[-1]
        X = b[:, :, : (size - 1)]
        y = b[:, :, -1]    
        cond["symbolic_conditioning"] = cond["symbolic_conditioning"].unsqueeze(0)  
        cond["numerical_conditioning"] = cond["numerical_conditioning"].unsqueeze(0) 
        outputs = fitfunc(X, y, cond, is_batch=True)

        print('True equation: ', inputs[2][0][0])
        print('Predicted equation: ', outputs['best_pred'])
    
    


if __name__ == '__main__':
    main()