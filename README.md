# CS279 - Reaction-Diffusion Simulation

## About

A reaction diffusion simulator for a Stanford CS279 final project (Autumn 2019). Our implemented
`Simulation` class allows user-defined simulations of Gray-Scott reaction-diffusion models as well
as custom stochastic reaction-diffusion simulations based on reaction kinetics.
 

## General Project Structure
```
├── run_simulation.py
├── requirements.txt
├── src
│   ├── VectorizedSimulation.py
│   ├── preset_simulations.py
│   └── config.py
└── simulations
    ├── example_1
    ├── example_2
    └── [...] <other simulation results>
```

## Requirements

Example virtual environment setup (using virtualenv):
```bash
pip install virtualenv  # get virtualenv if you don't have it already
virtualenv venv  # create virtualenv named 'venv'
source venv/bin/activate # activate this virtual environment
```

Install dependency requirements using the following command:
```bash
pip install -r requirements.txt  # install the requisite files
```

## Example Simulations

Running simulations using the `run_simulation.py` wrapper function.
Pre-designed simulations are provided in `src/preset_simulations.py`
and can be run using command-line arguments for `run_simulation.py`.

For example, running the base code with no additional parameters will
run a two-particle Gray-Scott simulation with `feed=0.0362` and `kill=0.062`. 

```bash
python run_simulation.py
```

You can run another pre-set simulation `feed=0.03` and `kill=0.062` by
running:

```bash
python run_simulation.py --type dots
```

By default, simulation results will be stored in `./simulations/test/...`.
To name the directory something else, simply provide the `--run_name <name>` tag.

In general, there are many user-modifiable parameters that can be accessed
through command line flags like this one. 

```bash
python run_simulation.py \
--run_name mySimulation \
--feed 0.035 \
--kill 0.062 \
--updates_per_frame 25 \
--iterations 1000
```

You can always run the `--help` flag to learn more.
