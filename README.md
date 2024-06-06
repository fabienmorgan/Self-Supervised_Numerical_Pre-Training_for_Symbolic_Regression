# Self-Supervised Numerical Pre-Training for Symbolic Regression 
This repository contains the code and resources for the bachelor thesis "Self-Supervised Numerical Pre-Training for Symbolic Regression" by Fabien Morgan.

### Overview
This Bachelor thesis investigates recent advancements in the field of Neural Symbolic Regression (NSR) and applies these methods. The novel approach utilized for NSR in this thesis is Contrastive Learning, inspired by the paper "MMSR: Symbolic Regression is a Multimodal Task" (arXiv:2402.18603). The performance of the models was extensively analyzed, particularly considering domain shifts. The code is built upon the [codebase](https://github.com/SymposiumOrganization/ControllableNeuralSymbolicRegression/tree/main) of the paper "Controllable Neural Symbolic Regression" (arXiv:2304.10336).


## Getting Started

### Prerequisites
* Tested on Python 3.11 
* Tested on Ubuntu 20.04
* Tested on PyTorch 2.2.1 
* Tested on Pytorch Lightning 1.9.5
### Installation
Clone the repository:
``` 
git clone https://github.com/fabienmorgan/Self-Supervised_Numerical_Pre-Training_for_Symbolic_Regression.git
```
Create and source the virtual environment:
```
python3 -m venv env
source env/bin/activate
```
Install PyTorch from https://pytorch.org/get-started, version 2.2.1 is greatly recommended.
Install the ControllableNesymres package and its dependencies:
```
cd src/
pip install -e .
```
Download the weights of the trained model and place them in the model folder:
```
cd weights/
```

Please check the requirements file if you encounter trouble with some other dependencies.

## Test on own equations
1. Download the weights from HuggingFace (model "Epoch_215_SEncoder_5_new_loss.ckpt" is the final implementation of the project and the best model):
```
git clone https://huggingface.co/fabien-morgann/Self-Supervised-Numerical-Pre-Training-for-Symbolic-Regression 
```  

2. If you want to reproduce the experiment on your own equations, you will need to convert from the csv format into the dataloader format. To do so, run the following script:
```
scripts/data_creation/convert_csv_to_dataload_format.py 
```
This script will create a new folder called "benchmark" inside the data folder. Inside this folder, it will create a folder for each benchmark set. For the thesis only the dataset `train_nc` was used. The models where not tested on the other dataset and for datasets with constants it is likely that further modifications need to be done.

3. Run the test:
```
python3 scripts/test.py
Arguments:
    --min_support: The minimum support range of the sampled data points
    --max_support: The maximum support range of the sampled data points
    --model_type: The name of the model (There is a dictionary with the names in test.py if you have safed the model in the default path)
    --test_path: The path of the train_nc dataset
    --seed: The random seed for reproducability
    --number_of_samples: Number of sampled data points for each to predict equation
    --custom_model_path: If you don't safe the model in the default path you can set where the checkpoint file is safed
    --skeleton_encoder_layers: If you use a custom model path then the number of skeleton encoder layers need to be defined
```
4. The evaluation of the test with the predicted equation, the true equation and further information can be found in the `evalutaion/` folder.

5. If multiple tests are done they can be agregated with the script `scripts/evaluate_test.py`. The script will create a CSV file with the aggrated evaluation under `evaluation/evaluation_summary.csv`. If the same run has been done twice on accident the duplicate run can be deleted with the script `scripts/delelte_duplicate_eval_files.py`.

6. The folder `evaluation_graphs/` has all the visualisations done for the thesis and additionally has a notebook that checks if the aggregated file of `evaluation/evaluation_summary.csv` has all the required content for the visualisations. It is important to note that to do all the evaluations that the thesis did it takes multiple days on a RTX 3080.

## Reproducing the training
### Data Generation (Training)
Generate synthetic datasets using the data_generation module. For the experiments 10 million equations where used with the following parameters:
``` 
python3 scripts/data_creation/create_dataset.py  --root_folder_path target_folder  --number_of_equations 10000000 --cores 32 --resume
Arguments:
    --root_folder_path: path to the folder where the dataset will be saved
    --number_of_equations: number of equations to generate
    --cores: number of cores to use for parallelization
Additional Arguments:
    --resumed: if True, the dataset generation will be resumed from the last saved file
    --eq_per_block: number of equations to generate in each block
``` 
Note that:
1. With 32 cores it took around 2 days to generate the 10M dataset. 
2. In some cases the data generation process could hang. In this case, you can kill the process and resume the generation from the last saved file using the --resume flag.
3. Equations are generated randomly by a seed dependent on the current date and time, so you will get different equations every time you run the script. If you want to generate the same equations, you can set the seed manually in the script (line 754 of src/ControllableNesymres/dataset/generator.py)

Optionally, after generating the dataset you can remove equations that are numerically meaningless (i.e. for all x in the range, the equation is always 0, infinity, or NaN) using the following scripts:
1. Identify the equations to remove. This script will create a npy containing for equation a tuple (equation_idx, True/False) where False means that the equation is invalid. This file is called equations_validity.npy and is saved in the same folder as the dataset.
```
python3 scripts/data_creation/check_equations.py --data_path target_folder/the_dataset_folder
Arguments:
    --data_path: Path to the dataset created with create_dataset.py
Additional Arguments:
    --debug/--no-debug: if --no-debug, the script is run with multiprocessing
```
2. Create a new dataset with only the good equations. This script will create a new dataset in the same folder as the original one, but inside
a folder called "datasets" instead of "raw_datasets".
```
python3 scripts/data_creation/remove_invalid_equations.py --data_path target_folder/the_dataset_folder
```
Arguments:
    --data_path: Path to the dataset created with create_dataset.py
Additional Arguments:
    --debug/--no-debug: if --no-debug, the script is run with multiprocessing

### Model Training
Train the model using the model module:
``` 
python scripts/train.py host_system_config.train_path=target_folder/datasets/10000000 
``` 
Note we make use of [Hydra](https://hydra.cc) to manage the configuration. The associated configuration file is located in scripts/config.py. You can change the configuration by either editing the file or by passing the desired parameters as command line arguments. For example, to train the model with a different number of epochs you can run:
```
python scripts/train.py  host_system_config.train_path=target_folder/datasets/10000000 host_system_config.batch_size=100
```
Take a look at the configuration file for more details about the available parameters.

## Host Configuration
Because the training hardware is individual the host configuration is seperate to the config. This means that a folder needs to be created in the path of the config named `host_system_config` and in this folder a file named `host.yaml` needs to be created. This file only contains host specific files and this is done so that it is possible to work on the repository on multiple different computer with different hardware specification and files safed in different paths. The file is  not checked in and needs to be manually created. The host config needs these seven parameters:  
`train_path`  
`benchmark_path`  
`model_path`  
`num_of_workers`  
`batch_size`  
`precision`  
`accelerator`  
`accelerator_devices`  
`resume_from_checkpoint`  
`path_to_candidates` 

An example of this file could look like this (host.yaml):  
``` yaml
# Need to point to the correct paths
train_path: training_dataset/raw_datasets/1000
benchmark_path: test_set

### Test (Not influence the training)
model_path: run/False/2022-11-07/13-46-03/Exp_weights/1000000_log_-epoch=104-val_loss=0.00.ckpt
### 

num_of_workers: 6
batch_size: 50 # 50 ideal if gpu(rtx 3080) is empty 
precision: 16
accelerator: "gpu" # Use gpu 
accelerator_devices: 1 # Number of the gpu to use
resume_from_checkpoint: ""

path_to_candidate: configs/equations_ops_3_5000.json # This is the file that contains the negative equations from which the model will sample the absent branches 
```


## License
This project is licensed under the MIT License
