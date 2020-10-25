# Model-based Reinforcement Learning for Semi-Markov Decision Processes with Neural ODEs

This repository is the official implementation of "Model-based Reinforcement Learning for Semi-Markov Decision Processes with Neural ODEs", NeurIPS 2020 [[arxiv](https://arxiv.org/abs/2006.16210)].

## Requirements

We assume the common scientific computing libraries, e.g., numpy, scipy are installed by Anaconda for Python 3.6.7. We are using PyTorch 1.3.1 for implementation. To install PyTorch 1.3.1, run this command:

```
conda install pytorch==1.3.1 torchvision==0.4.2 -c pytorch
```

To install other required libraries for this project, run this command:

```
pip install torchdiffeq gym 'mujoco-py<2.1,>=2.0'
```

Note that [Mujoco](http://www.mujoco.org/index.html) is a commercial software which requires a license for installation. Check out [how to install mujoco-py](https://github.com/openai/mujoco-py) for details.


## Usage

To run the experiment on the windy gridworld, acrobot and HIV environments, run this command (take Latent-ODE and HIV for example, check different hyperparameters for different tasks in the supplement):

```
python run.py --train_env_model --world_model --model latent-ode --env hiv --timer mlp --iters 12000 --trajs 1000 \
       --gamma 0.995 --lr 0.001  --batch_size 32 --eps_decay 0 --latent_dim 10 --max_steps 50 --mem_size 100000 \
       --episodes 1500 --ode_dim 20 --ode_tol 1e-3 --enc_hidden_to_latent_dim 20 --num_restarts 1 --seed 1 --log
```

To run the experiment on Mujoco tasks, run this command (take Latent-ODE and HalfCheetah for example, check different hyperparameters for different tasks in the supplement):

```
python run.py --mpc_ac --obs_normal --model latent-ode --env half_cheetah --timer mlp --epochs 200 --gamma 0.99 \
       --lr 0.001  --batch_size 128 --eps_decay 0 --latent_dim 400 --max_steps 1000 --mem_size 1000000 --mb_epochs 80 \
       --mf_epochs 0 --env_steps 5000 --planning_horizon 10 --ode_dim 400 --ode_tol 1e-5 --enc_hidden_to_latent_dim 20 \
       --num_restarts 1 --seed 1 --log
```


## Pre-trained Models

We provide pre-trained Latent-ODEs for all domains in [pretrain](pretrain). Note that for the windy gridworld, acrobot and HIV environments, models are trained on a fixed dataset, and for Mujoco tasks, models are trained with newly and previously collected data every epoch.

## Bibtex
```
@article{du2020model,
  title={Model-based Reinforcement Learning for Semi-Markov Decision Processes with Neural ODEs},
  author={Du, Jianzhun and Futoma, Joseph and Doshi-Velez, Finale},
  journal={arXiv preprint arXiv:2006.16210},
  year={2020}
}
```
