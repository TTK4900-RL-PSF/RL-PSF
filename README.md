# gym-turbine
A repository for TTK4900 Master thesis at NTNU. Project is stabilizing a floating off-shore wind turbine using Reinforcement Learning and a PSF

Written in `Python 3.7`.
## Installation
### Mosek
You will need a licence to use Mosek (free academic):\
https://www.mosek.com/license/request/?i=acp

Add the license file to `%USERPROFILE%/mosek/mosek.lic`.

This is only required when generating a new terminal set.

### Python dependencies and packages

Conda env is highly recommended due to import errors with pip.

Run (w/ conda):

```
conda env create -f environment.yml
conda activate gym-rl-mpc
```
Alternatively, dependencies are listed in
`environment.yml` and `setup.py`.


## Running the program


### Training
To train an agent run:
```
python train.py
```
Optional arguments:
- --agent <path to pretrained agent .zip file to start training from>
- --timesteps <number of timesteps to train the agent>
- --env <environment to run (e.g. VariableWindLevel3-v17)>
- --psf <use PSF corrected actions>
- --note <Note with additional info about training, gets added to Note.txt>
- --no_reporting <Skip reporting>

To view the tensorboard logs run (in another cmd window):
```
tensorboard --logdir logs
```

NOTE: The environment config is located in gym_rl_mpc/\_\_init\_\_.py

### Simulating with a trained agent
To simulate the system with an agent run:
```
python run.py
```
Required arguments:
- --agent <path to agent .zip file>

Optional arguments:
- --time <Max simulation time (seconds)>
- --plot
- --env <environment to run (e.g. VariableWindLevel3-v17)>
- --psf <use PSF corrected actions>


### Animating
To animate a simulation of an agent, run:
```
python animate.py
```
Required arguments:
- --agent <path to agent .zip file>

Optional arguments:
- --save_video
- --time <Max simulation time (seconds)>
- --env <environment to run (e.g. VariableWindLevel3-v17)>
- --psf <use PSF corrected actions>

Or to show an animation of a simulation run (from file):
```
python animate.py
```
Required arguments:
- --data <path to .csv data file>

Optional arguments:
- --save_video
- --time <Max simulation time (seconds)>


## Based on

https://arxiv.org/pdf/1812.05506.pdf
