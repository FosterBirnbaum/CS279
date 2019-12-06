# CS279

### General Project Structure
```
├── VectorizedSimulation.py
├── run_simulation.py
├── preset_simulations.py
├── config.py
├── requirements.txt
└── simulations
    ├── example_1
    ├── example_2
    └── [...] <other simulation results>
```

## Requirements

Virtual environment setup
```bash
pip install virtualenv  # get virtualenv if you don't have it already
virtualenv venv  # create virtualenv named 'venv'
source venv/bin/activate # activate this virtual environment
```

```bash
pip install -r requirements.txt  # install the requisite files
```

## Example Run

Running the default script will simulate a run of two particles with
`feed=0.0545`, `kill=0.03`, and `length=100`.

It will save the results under 
`simulations/simple_iterations-100_length-100_feed-0.0545_kill-0.03`

```bash
python SimpleGrayScott.py
```
