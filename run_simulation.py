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

from src.VectorizedSimulation import Simulation, open_file
import src.preset_simulations as preset
import src.config as config
import argparse
import glob
import os

def main(args):

    # =======================================
    # || Pre-defined simulation parameters ||
    # =======================================

    # Standard two-particle Gray-Scott model
    if args.type[0] == "gray_scott":
        sim_params = preset.get_gray_scott_config(preset.default)
    elif args.type[0] == "dots":
        sim_params = preset.get_gray_scott_config(preset.dots)
    elif args.type[0] == "waves":
        sim_params = preset.get_gray_scott_config(preset.waves)
    elif args.type[0] == "circles":
        sim_params = preset.get_gray_scott_config(preset.circles)
    elif args.type[0] == "first_order":
        sim_params = preset.first_order
    elif args.type[0] == "second_order":
        sim_params = preset.second_order
    elif args.type[0] == "cellular_open":
        sim_params = preset.cellular_open
    elif args.type[0] == "cellular_restricted":
        sim_params = preset.cellular_restricted
    else:
        raise Exception("Unrecognized simulation type. %r" % args.type[0])

    # =======================================
    # || Manual Over-ride of parameters    ||
    # =======================================
    sim_params["feed"] = args.feed if args.feed is not None else sim_params["feed"]
    sim_params["kill"] = args.kill if args.kill is not None else sim_params["kill"]
    sim_params["length"] = args.length if args.length is not None else sim_params["length"]
    sim_params["updates_per_frame"] = args.updates_per_frame if \
        args.updates_per_frame is not None else sim_params["updates_per_frame"]

    # Create simulation object using parameters defined above
    sim = Simulation(**sim_params)

    # Run the selected simulation and save to the results to simulations\[<input name>]-[<params>]
    sim.run(iterations=args.iterations, run_name=args.run_name, visual=args.visual)

    # Run the simulation video after compilation
    video_path = glob.glob(os.path.join(sim.output_path, '*.avi'))[0]
    open_file(video_path)


if __name__ == "__main__":

    # Get user input via command-line arguments
    parser = argparse.ArgumentParser(description='Run a diffusion simulation model.')

    parser.add_argument('--type', type=str, nargs=1,
                        choices=["gray_scott",
                                 "dots",
                                 "waves",
                                 "circles",
                                 "first_order",
                                 "second_order",
                                 "cellular_open",
                                 "cellular_restricted"],
                        default=["gray_scott"],
                        help='Select a pre-designed simulation type to run')
    parser.add_argument('--run_name', type=str, default="test",
                        help='Enter a base prefix run name for the saved simulation directory.')

    parser.add_argument('--iterations', type=int, default=200,
                        help='Enter the number of iterations for which to run the simulation.')
    parser.add_argument('--length', type=int, help='Simulation dimension.')
    parser.add_argument('--visual', dest='visual', action='store_true',
                        help='Whether or not to visualize the simulation real-time.')
    parser.set_defaults(visual=False)
    parser.add_argument('--feed', type=float, help='Feed rate.')
    parser.add_argument('--kill', type=float, help='Kill rate.')
    parser.add_argument('--updates_per_frame', type=int, help='Num simulation time steps / frame')


    args = parser.parse_args()

    main(args)