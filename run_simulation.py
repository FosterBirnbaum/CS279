"""
File: run_simulation.py
=======================
Main entry point for Simulation class.
- Default run will run a standard two-particle Gray-Scott simulation with
    (feed, kill) = (0.0362, 0.062). Simulation will be run for 200 frames
    with 50 simulation updates per frame.
- Using command-line arguments, the behaviour of the simulation can be
    dramatically changed.
- Run `python3 run_simulation --help` for information about each command
    flag

Example:
    `python3 run_simulation --type two_particles --feed 0.05 --kill 0.06\
        --iterations 1000 --updates_per_frame 50 --run_name my_simulation`

    Will run a two-particle Gray-Scott simulation with (feed, kill) = (0.05,0.06)
    for 1000 frames at 50 updates per frame. It will save simulation results
    to a './simulations/my_simulation' directory.

Notes:
- By default, the run directory is 'test' which will be overwritten upon each
    simulation run. This is to make running multiple test simulations easy
    and not cumbersome.
"""

from VectorizedSimulation import Simulation, open_file
import config
import argparse
import glob
import os

def main(args):

    # Pre-defined simulation parameters
    # Standard two-particle Gray-Scott model
    if args.type[0] == "two_particles":
        n = 2
        orders = [-1, -1]
        diffusions = [1, 0.5]
        feed = args.feed
        kill = args.kill
        temp = None
        length = args.length
        init = "pointMass"
        activationEnergies = None
        startingConcs = [0.25, 0.25, 0]
        laplace_matrix = config.DEFAULT_LAPLACE_MATRIX

    # Three particle reactions controlled by first-order kinetics
    # A + B --> C
    elif args.type[0] == "three_particles_first_order":
        n = 3
        orders = [1, 1, 1]
        diffusions = [1, 1, 0.5]
        feed = None
        kill = None
        activationEnergies = [4, 4]
        temp = 350
        length = args.length
        init = "even"
        startingConcs = [0.5, 0.25, 0]
        laplace_matrix = config.DEFAULT_LAPLACE_MATRIX

    # Three particle reactions controlled by second-order kinetics
    # 2A + B --> C
    elif args.type[0] == "three_particles_second_order":
        n = 3
        orders = [2, 2, 2]
        diffusions = [1, 1, 0.5]
        feed = None
        kill = None
        activationEnergies = [3, 3]
        temp = 298
        length = args.length
        init = "even"
        startingConcs = [0.5, 0.25, 0]
        laplace_matrix = config.DEFAULT_LAPLACE_MATRIX

    # Normal diffusion about a cell for three-particle
    # 2A + B --> C
    elif args.type[0] == "cellular_open":
        n = 3
        orders = [1, 1, 1]
        diffusions = [1, 1, 0.5]
        feed = None
        kill = None
        activationEnergies = [3, 3]
        temp = 350
        length = args.length
        init = "cellular-open"
        startingConcs = [0.5, 1, 0]
        laplace_matrix = config.DEFAULT_LAPLACE_MATRIX

    elif args.type[0] == "cellular_restricted":
        n = 3
        orders = [1, 1, 1]
        diffusions = [1, 1, 0.5]
        feed = None
        kill = None
        activationEnergies = [3, 3]
        temp = 350
        length = args.length
        init = "cellular-restricted"
        startingConcs = [0.5, 1, 0]
        laplace_matrix = config.RESTRICTED_LAPLACE_MATRIX

    else:
        raise Exception("Unrecognized simulation type. %r" % args.type[0])

    # Create simulation object using parameters defined above
    sim = Simulation(n=n, orders=orders, diffusions=diffusions, feed=feed, kill=kill, length=length,
                     init=init, activationEnergies=activationEnergies, startingConcs=startingConcs,
                     laplace_matrix=laplace_matrix, temp=temp)

    # Run the selected simulation and save to the results to simulations\[<input name>]-[<params>]
    sim.run(iterations=args.iterations, run_name=args.run_name, updates_per_frame=args.updates_per_frame,
            visual=args.visual)

    # Run the simulation video after compilation
    video_path = glob.glob(os.path.join(sim.output_path, '*.avi'))[0]
    open_file(video_path)


if __name__ == "__main__":

    # Get user input via command-line arguments
    parser = argparse.ArgumentParser(description='Run a modified Gray-Scott diffusion simulation model.')

    parser.add_argument('--type', type=str, nargs=1,
                        choices=["two_particles", "three_particles_first_order",
                                 "three_particles_second_order", "cellular_open", "cellular_restricted"],
                        default=["two_particles"],
                        help='Select a pre-designed simulation type to run')
    parser.add_argument('--run_name', type=str, default="test",
                        help='Enter a base prefix run name for the saved simulation directory.')
    parser.add_argument('--iterations', type=int, default=200,
                        help='Enter the number of iterations for which to run the simulation.')
    parser.add_argument('--feed', type=float, default=0.0362,
                        help='Feed rate.')
    parser.add_argument('--kill', type=float, default=0.062,
                        help='Kill rate.')
    parser.add_argument('--length', type=int, default=50,
                        help='Simulation dimension.')
    parser.add_argument('--updates_per_frame', type=int, default=50,
                        help='Number of updates of simulation to perform for each frame of the animation.')
    parser.add_argument('--visual', type=bool,
                        help='Whether or not to visualize the simulation real-time.')

    args = parser.parse_args()

    main(args)