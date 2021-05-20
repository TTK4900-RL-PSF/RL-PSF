import argparse
import hashlib
import os
import re
import shutil
import subprocess
from pathlib import Path
"""
ENVS = [
    'VariableWindLevel0-v17',
    'VariableWindLevel1-v17',
    'VariableWindLevel2-v17',
    'VariableWindLevel3-v17',
    'VariableWindLevel4-v17',
    'VariableWindLevel5-v17'
]
"""
ENVS = [
    'ConstantWindLevel1-v17',
    'ConstantWindLevel2-v17',
]

SBATCH_DIR = Path("sbatch")

SBATCH_CONFIG = ("#!/bin/sh\n"
                 "#SBATCH -J RL_w_PSF\n"  # Sensible name for the job"
                 "#SBATCH -N 1\n"  # Allocate 2 nodes for the job"
                 "#SBATCH --ntasks-per-node=1\n"  # 1 task per node"
                 f"#SBATCH -c {12}\n"
                 "#SBATCH -t 24:00:00\n"  # days-hours:minutes:seconds Upper time limit for the job"
                 "#SBATCH -p CPUQ\n")

MODULES = "module load fosscuda/2020b\n"

CONDA_HACK = "source /cluster/apps/eb/software/Anaconda3/2020.07/etc/profile.d/conda.sh\n"
CONDA_ACT = "conda activate gym-rl-mpc\n"


def create_run_files(
        run_list,
        sbatch_config=SBATCH_CONFIG,
        modules=MODULES,
        sbatch_dir=SBATCH_DIR
):
    common_setup = sbatch_config + modules + CONDA_HACK + CONDA_ACT

    try:
        shutil.rmtree(sbatch_dir)
    except FileNotFoundError:
        pass

    os.mkdir(sbatch_dir)

    for run_str in run_list:
        text = common_setup + run_str
        filename = hashlib.md5(text.encode()).hexdigest()
        with open(Path(sbatch_dir, filename + '.sh'), 'w') as f:
            f.write(text)


def create_train_files():
    parser = argparse.ArgumentParser("Python implementation of spawning a lot of sbatches on train."
                                     "The env currently to run are ENVS. Queues with and without PSF")

    parser.add_argument(
        '--num_cpus',
        type=int,
        default=12,
        help='Number of cores to ask for. Default is 12 with good reason'
    )
    parser.add_argument(
        '--timesteps',
        type=int,
        default=10000000,
        help='Number of timesteps to train the agent. Default=10000000',
    )
    parser.add_argument(
        '--agent',
        help='Path to the RL agent to continue training from.',
    )
    parser.add_argument(
        '--note',
        type=str,
        default=None,
        help="Note with additional info about training"
    )
    parser.add_argument(
        '--no_reporting',
        help='Skip reporting to increase framerate',
        action='store_true'
    )
    parser.add_argument(
        '--psf_T',
        type=int,
        default=10,
        help='psf horizon'
    )
    args = parser.parse_args()
    s_config = ("#!/bin/sh\n"
                "#SBATCH -J RL_w_PSF\n"  # Sensible name for the job"
                "#SBATCH -N 1\n"  # Allocate 2 nodes for the job"
                "#SBATCH --ntasks-per-node=1\n"  # 1 task per node"
                f"#SBATCH -c {args.num_cpus}\n"
                "#SBATCH -t 01-10:00:00\n"  # days-hours:minutes:seconds Upper time limit for the job"
                "#SBATCH -p CPUQ\n")

    opts = ""
    for k, v in args.__dict__.items():
        opts += f" --{k} {v}" if v else ''

    run_list = []
    for psf_opt in ["", " --psf"]:
        for env in ENVS:
            run_list += [f'python train.py --env {env}' + psf_opt + opts]

    create_run_files(run_list, s_config)


def bat_to_sbatch(bat_dir="batfolder", pattern=".*?gen_.*psf"):
    files = [f.name for f in os.scandir(bat_dir) if f.is_file()]
    bat_gen_files = []
    for s in files:
        if re.search(pattern, s):
            bat_gen_files.append(s)
    run_commands = []

    for bat_gen in bat_gen_files:
        with open(Path(bat_dir, bat_gen)) as f:
            run_commands += [s for s in f.readlines() if "python" in s]

    s_config = ("#!/bin/sh\n"
                "#SBATCH -J bat_run\n"  # Sensible name for the job"
                "#SBATCH -N 1\n"  # Allocate 2 nodes for the job"
                "#SBATCH --ntasks-per-node=1\n"  # 1 task per node"
                f"#SBATCH -c {1}\n"
                "#SBATCH -t 12:00:00\n"  # days-hours:minutes:seconds Upper time limit for the job"
                "#SBATCH -p CPUQ\n")

    create_run_files(run_commands, s_config)


def call_sbatch(batch_loc=SBATCH_DIR):
    filenames = os.listdir(batch_loc)
    for f in filenames:
        print(Path(batch_loc, f).absolute())
        subprocess.run(["sbatch", str(Path(batch_loc, f))])


if __name__ == '__main__':
    create_train_files()
    call_sbatch()

