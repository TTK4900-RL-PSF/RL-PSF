
import os
from itertools import repeat
from multiprocessing import Pool
from pathlib import Path
import run

N_ITER = 10
time = 100

HERE = Path(__file__).parent
os.chdir(HERE.parent)


if __name__ == '__main__':
    # start 6 worker processes
    args = vars(run.parse_argument())
    args["time"] = time
    args["psf"] = True
    print(f"Running {N_ITER} number of iterations, each {time} s through 'run.py'")
    number_of_workers = 6

    print(f"Spinning up {number_of_workers} processes")
    run.run(args)
    with Pool(processes=6) as pool:
        pool.map(run.run, repeat(args, N_ITER))

    print("\n\n**** No crash reported ****\n\n")
