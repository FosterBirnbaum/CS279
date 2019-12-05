"""
This file defines the particle and simulation classes that allow users
to run a variety of simulation types. The available types are as follows:
    -TwoParticleGS: Simulates normal Gray-Scott (2B + A --> 3B, feed rate for A, kill rate for B)
    -ThreeParticleFirstOrder: Simulates a reaction involving 3 particles (A + B --> C) that is first order in A and B
    -ThreeParticleSecondOrder: Simulates a reaction involving 3 particles (A + B --> C) that is second order in A and first order in B
    -ThreeParticleFirstOrderWithCatalyst: Simulates a reaction involving 3 particles (A + B --> C) that is first order in substrates and occurs in presence of catalyst
    -Custom: Allows user to define input parameters
All simulations are saved in video format, as are plots of relative concentrations of all species involved and a plot of the rate of formation of product (if one is involved).

Written by Collin Schlager (schlager@stanford.edu) and Foster Birnbaum (fosb@stanford.edu) -- 12/6/2019
The backbone of the particle class was written by instructors of CS279 in Fall 2019
"""

from tqdm import tqdm
from matplotlib import pyplot as plt
import subprocess
import numpy as np
from scipy import ndimage
import scipy.stats as stats
import os
import glob
import sys
import logging
import shutil
import argparse
import matplotlib.animation as animation
import config
np.set_printoptions(precision=2)

"""
Defines the particle class that contains information on a specific particle in the simulation.
Each particle is defined as an nX by nY grid, with values in each grid box representing the 
concentration of that particle at that position.
"""
class Particle(object):
    def __init__(self, nX=25, nY=25, diffusion=1.0):
        self.nX = nX
        self.nY = nY
        self.blocks = np.zeros((nX, nY), dtype=np.float64)
        self.diffusion = diffusion


def open_file(filename):
    if sys.platform == "win32":
        os.startfile(filename)
    else:
        opener = "open" if sys.platform == "darwin" else "xdg-open"
        subprocess.call([opener, filename])


"""
Defines the Simulation object that runs a reaction-diffusion simulation according
to user input parameters. For a list of sample input parameters, please see top of file comment.
"""
class Simulation(object):
    def __init__(self, n=2, orders=[-1, -1], diffusions=[1, 1], feed=0.0545, kill=0.03, activationEnergies=[1, 1],
                 temp=298, length=100, maxConc=3, startingConcs=config.STARTING_CONCS,
                 laplace_matrix=config.DEFAULT_LAPLACE_MATRIX, init=config.DEFAULT_INIT):
        """
        Object to simulate Gray-Scott diffusion with n particles.
        :param n: number of particles
        :param orders: array of reaction orders for each substrate molecule
                       if the order is -1, the default Gray-Scott equation is used instead of an equation based on reaction kinetics
        :param diffusions: array of diffusion constants for each molecule
        :param feed: feed rate
        :param kill: kill rate
        :param activationEnergies: barriers to the reaction occuring
        :param temp: temperature at which the reaction occurs
        :param length: side-length of simulation square
        :param maxConc: max concentration
        :param laplace_matrix: 3x3 convolution matrix
        :param intit: string identifying how grid should be initialized
        :return None
        """

        # Class parameters
        self.numParticles = n
        self.orders = orders
        self.length = length
        self.maxConc = maxConc
        self.startingConcs = startingConcs
        self.feed = feed
        self.kill = kill
        self.activationEnergies = activationEnergies
        self.temp = temp
        self.laplace_matrix = laplace_matrix
        self.laplacians = np.zeros((n, length, length), np.float64)
        self.init = init

        # Init particles
        self.particleList = []
        for i in range(n):
            self.particleList.append(Particle(nX=length, nY=length, diffusion=diffusions[i]))

        # Set initial particle state
        self.set_initial_state()

    def set_initial_state(self):
        """
        Sets the initial state of the particle objects for the simulation.
        TODO: make this more dynamic
        """
        # If user wants even intialization, put first two particles to starting concentration everywhere
        if self.init == "even":
            for particle in range(self.numParticles - 1):
                for i in range(self.length):
                    for j in range(self.length):
                        self.particleList[particle].blocks[i, j] = self.startingConcs[particle]
        # If user wants seperated initialization, put first particle to starting concentration on one half and second to starting concentration on other half (with small gap in between)
        elif self.init == "seperated":
            for i in range(self.length):
                for j in range(self.length):
                    if (i < self.length * 0.5 - self.length * 0.05):
                        self.particleList[0].blocks[i, j] = self.startingConcs[0]
                    elif (i > self.length * 0.5 + self.length * 0.05):
                        self.particleList[1].blocks[i, j] = self.startingConcs[1]
        # If user wants cellular initialization, establish a region in the center with first particle present and from which it cannot diffuse out of, and set
        # second particle to a very high concentration elsewhere in the grid
        elif "cellular" in self.init:
            for i in range(self.length):
                for j in range(self.length):
                    if (self.length * 0.5 - self.length * 0.06 <= i <= self.length * 0.5 + self.length * 0.06 and
                            self.length * 0.5 - self.length * 0.06 <= j <= self.length * 0.5 + self.length * 0.06):
                        self.particleList[0].blocks[i, j] = self.startingConcs[0]
                    elif (i < self.length * 0.1) and (j < self.length * 0.1):
                        self.particleList[1].blocks[i, j] = self.startingConcs[1]
        # If user wants random intialization, randomly assign (with mean of starting concentration) concentration of first two particled everywhere
        elif self.init == "random":
            for particle in range(self.numParticles - 1):
                for i in range(self.length):
                    for j in range(self.length):
                        self.particleList[particle].blocks[i, j] = np.random.normal(self.startingConcs[particle], 0.5)
        # If user wants point mass, put first particle to 1 everywhere and second particle to 0 everywhere except a small region in center
        elif self.init == "pointMass":
            for i in range(self.length):
                for j in range(self.length):
                    self.particleList[0].blocks[i, j] = 1
                    if (self.length * 0.5 - i) ** 2 + (self.length * 0.5 - j) ** 2 <= self.length * 0.05:
                        self.particleList[1].blocks[i, j] = 1
                        self.particleList[0].blocks[i, j] = 0
                    else:
                        self.particleList[1].blocks[i, j] = 0

    def run(self, iterations, run_name="simple", updates_per_frame=25, visual=False):
        """
        Runs the simulation and stores frames to a result directory in `simulatons/`
        :param iterations: number of iterations to run the simulation for
        :param using: which graphics library to use ["matplotlib" or "cvs"]. Default is "cv2"
        :param run_name: name of the simulation run. Used as prefix for output directory
        :return: None
        """

        # Create and go to output directory
        output_dir_name = run_name

        output_path = os.path.join(config.OUTPUT_FOLDER, output_dir_name)
        self.output_path = output_path

        if run_name == "test":
            if os.path.exists(output_path):
                shutil.rmtree(output_path)

        if os.path.exists(output_path):
            logging.error("Simulation already exists with same name and parameters. "
                          + "Please change run_name or delete the existing directory.")
            return
        else:
            cur_dir = os.getcwd()
            os.mkdir(output_path)
            os.chdir(output_path)

        # Run the proper simulation
        self.run_matplotlib(iterations, run_name, updates_per_frame=updates_per_frame, visual=visual)

        # Change back to the original directory
        os.chdir(cur_dir)

    def run_matplotlib(self, iterations, run_name, updates_per_frame=25, visual=False):
        """
        Use matplotlib as graphics library to create a video of the simulation.
        :param iterations: Number of iterations for simulation
        :param run_name: Prefix for run files ("<run_name>-<output_type>")
        :param save_frames: If true, saves individual frames frame files ("<run_name>-<output_type>.png")
        :return: None
        """

        self.updates_per_frame = updates_per_frame

        # Start creating video
        metadata = dict(title='Movie Test', artist='Matplotlib', comment='Movie support!')
        fig = plt.figure()

        # Create numpy arrays to track total concentrations of all particles and rate
        # or product formation and figures to display concentrations and rate
        self.particleConcentrations = np.zeros((iterations, self.numParticles), np.float64)
        self.rate = np.zeros(iterations - 1, np.float32)
        self.timesteps = np.arange(iterations)
        self.rateTimesteps = np.arange(1, iterations)

        a = np.zeros((self.length, self.length))
        self.im = plt.imshow(a, interpolation='none', aspect='auto', animated=True, vmin=0, vmax=0.5)

        ani = animation.FuncAnimation(fig, self.get_frame, tqdm(range(iterations)),
                                      interval=50, blit=True, repeat_delay=1000)

        if visual:
            try:
                plt.show()

            except KeyboardInterrupt:
                pass

        ani.save(run_name + "-video.avi")
        plt.close()

        # Plot concentration and rate
        print("Creating plots...")
        colors = ['r', 'g', 'b']
        for i in range(self.numParticles):
            plt.plot(self.timesteps, self.particleConcentrations[:, i] / (self.length ** 2),
                     color=colors[i], label=('particle' + str(i)))
        plt.legend()
        plt.savefig(str(run_name) + '-concentrations.png')
        plt.close()

        plt.plot(self.rateTimesteps, self.rate, color=colors[0], label=('rate'))
        plt.legend()
        plt.savefig(str(run_name) + '-rate.png')
        plt.close()

        print("Complete.")

    def get_frame(self, frame):

        # perform multiple update steps per frame to speed up simulation
        for step in range(self.updates_per_frame):
            self.update()

        for i in range(self.numParticles):
            self.particleConcentrations[frame, i] = np.sum(self.particleList[i].blocks)

        # If past current frame, get rate of product formation
        if (frame > 0):
            self.rate[frame-1] = self.particleConcentrations[frame, -1] - self.particleConcentrations[frame - 1, -1]

        self.im.set_array(self.particleList[1].blocks)
        # plt.figure()
        # self.im = plt.imshow(self.particleList[1].blocks)
        # plt.savefig(str(frame) + '.png')
        # plt.close()
        return [self.im]

    def update(self):
        """
        Update step. Updates blocks of all types of particles using the framework input by user.
        :return: None
        """
        concentrationUpdates = self.get_particle_derivatives()
        for particle in range(self.numParticles):
            self.particleList[particle].blocks += concentrationUpdates[particle]
            # Ensure no values are above max or below min
            # (this shouldn't be an issue if update parameters are set appropriately)
            # np.where(self.particleList[particle].blocks>1, self.particleList[particle].blocks, 1)
            # np.where(self.particleList[particle].blocks<0, self.particleList[particle].blocks, 0)

    def compute_laplacians(self):
        """
        Compute the laplacians for all particles using the 3x3 convolution matrix.
        :return: None
        """

        for particle in range(self.numParticles):
            curGrid = self.particleList[particle].blocks

            # If init was set to cellular, then for the first and third particles
            # restrict movement to the center of the grid (i.e., the nucleus)
            # and for all other particles, do not allow movement outside the cell
            if "cellular" in self.init:
                if (particle == 0) or (particle == 2):
                    grid = self.particleList[particle].blocks
                    nucleus = grid[int(self.length*0.5 - self.length*0.06):int(self.length*0.5 + self.length*0.06), int(self.length*0.5 - self.length*0.06):int(self.length*0.5 + self.length*0.06)]
                    self.laplacians[particle,:,:] = np.zeros((self.length, self.length), np.float64)
                    self.laplacians[particle,int(self.length*0.5 - self.length*0.06):int(self.length*0.5 + self.length*0.06),int(self.length*0.5 - self.length*0.06):int(self.length*0.5 + self.length*0.06)] = ndimage.convolve(nucleus, self.laplace_matrix, mode='wrap')
                else:
                    #If user specific restricted cellular, use the restricted laplacian matrices
                    if "restricted" in self.init:
                        self.laplacians[particle,:,:] = ndimage.convolve(curGrid, config.RESTRICTED_LAPLACE_MATRIX,  mode='wrap')
                    else:
                        self.laplacians[particle,:,:] = ndimage.convolve(curGrid, self.laplace_matrix, mode='wrap')
            else:
                self.laplacians[particle, :, :] = ndimage.convolve(curGrid, self.laplace_matrix, mode='wrap')

    def compute_maxwell(self):
        """
        Compute the energy of the system according to the maxwell-boltzmann distribution
        :return: energy sampled from appropriate maxwell-boltzmann distribution
        """
        scale = 10 * ((self.temp - config.DEFAULT_TEMP_LOWERBOUND) /
                      (config.DEFAULT_TEMP_UPPERBOUND - config.DEFAULT_TEMP_LOWERBOUND))
        energy = stats.maxwell.rvs(loc=0, scale=scale, size=self.length**2)
        energy = energy.reshape(self.length, self.length)
        return energy

    def get_particle_derivatives(self):
        """
        Compute the change in all particles at a particular location (i, j). Returned equation is defined by
        value stored in orders array:
            -1: Gray-Scott
             0: Zero Order
             1: First Order
             2: Second Order
        :param i: x location to update
        :param j: y location to update
        :return: dParticledt evaluated at (i, j) for input particle
        """

        self.compute_laplacians()

        # Get starting concentrations and laplacian values
        conc_A = np.copy(self.particleList[0].blocks)
        conc_B = np.copy(self.particleList[1].blocks)
        conc_C = np.copy(self.particleList[2].blocks) if self.numParticles > 2 else 0
        lapA = self.laplacians[0, :, :]
        lapB = self.laplacians[1, :, :]
        lapC = self.laplacians[2, :, :] if self.numParticles > 2 else 0

        if (self.orders[0] == -1):

            dAdt = self.particleList[0].diffusion * lapA \
                   - np.multiply(conc_A, np.square(conc_B)) \
                   + self.feed * (1 - conc_A)

            dBdt = self.particleList[1].diffusion * lapB \
                   + np.multiply(conc_A, np.square(conc_B)) \
                   - (self.kill + self.feed) * conc_B

            dCdt = 0

        elif(self.orders[0] == 1):

            # masks for reactions happening
            curEnergy = self.compute_maxwell()
            react_AB = curEnergy > self.activationEnergies[0]
            react_C = curEnergy > self.activationEnergies[1]

            dAdt = self.particleList[0].diffusion*lapA
            dBdt = self.particleList[1].diffusion*lapB
            dCdt = self.particleList[2].diffusion*lapC
            conc_A += dAdt
            conc_B += dBdt
            conc_C += dCdt
            dAdt = dAdt - react_AB*conc_A*conc_B + react_C*conc_C
            dBdt = dBdt - react_AB*conc_A*conc_B + react_C*conc_C
            dCdt = dCdt + react_AB*conc_A*conc_B - react_C*conc_C

        elif(self.orders[0] == 2):

            curEnergy = self.compute_maxwell()
            react_AB = curEnergy > self.activationEnergies[0]
            react_C = curEnergy > self.activationEnergies[1]

            dAdt = self.particleList[0].diffusion*lapA
            dBdt = self.particleList[1].diffusion*lapB
            dCdt = self.particleList[2].diffusion*lapC
            conc_A += dAdt
            conc_B += dBdt
            conc_C += dCdt
            dAdt = dAdt - 2*(react_AB * conc_A**2 * conc_B) + 2*(react_C * conc_C)
            dBdt = dBdt - react_AB * conc_A**2 * conc_B + react_C * conc_C
            dCdt = dCdt + react_AB * conc_A**2 * conc_B - react_C * conc_C

        else:
            raise ValueError("Reaction order not recognized.")

        return [dAdt, dBdt, dCdt]


if __name__ == "__main__":

    # Get user input via command-line arguments
    parser = argparse.ArgumentParser(description='Run a modified Gray-Scott diffusion simulation model.')

    parser.add_argument('--type', type=str, nargs=1,
                        choices=["2Particles", "2ParticlesZeroOrder",
                                 "3ParticlesFirstOrder", "3ParticlesSecondOrder",
                                 "CellularOpen", "CellularRestricted"],
                        default=["2Particles"],
                        help='Select a pre-designed simulation type to run')
    parser.add_argument('--run_name', type=str, default=["test"],
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

    # Decide what type of simulation the user wants to run
    # Decide what type of simulation the user wants to run
    if args.type[0] == "2Particles":
        sim = Simulation(n=2, orders=[-1,-1], diffusions = [1, 0.5], feed = args.feed, kill = args.kill, length=args.length, init="pointMass")
    elif args.type[0] == "2ParticlesZeroOrder":
        sim = Simulation(n=2, orders=[0, 0], diffusions = [1, 0.5], activationEnergies = [3, 3], temp = 298, length=args.length, init="pointMass")
    elif args.type[0] == "3ParticlesFirstOrder":
        sim = Simulation(n=3, orders=[1,1,1], diffusions = [1, 1, 0.5], activationEnergies = [4, 4], temp = 350, length=args.length, init="even", startingConcs=[0.5, 0.25, 0])
    elif args.type[0] == "3ParticlesSecondOrder":
        sim = Simulation(n=3, orders=[2,2,2], diffusions = [1, 1, 0.5], activationEnergies = [3, 3], temp = 298, length=args.length, init="even", startingConcs=[0.5, 0.25, 0])
    elif args.type[0] == "CellularOpen":
        sim = Simulation(n=3, orders=[1,1,1], diffusions = [1, 1, 0.5], activationEnergies = [3, 3], temp = 350, length=args.length, init="cellular-open", startingConcs=[0.5, 1, 0])
    elif args.type[0] == "CellularRestricted":
        sim = Simulation(n=3, orders=[1,1,1], diffusions = [1, 1, 0.5], activationEnergies = [3, 3], temp = 350, length=args.length, init="cellular-restricted", startingConcs=[0.5, 1, 0], laplace_matrix=config.RESTRICTED_LAPLACE_MATRIX)
    else:
        # argparse should handle this, but just in case
        sim = None
        raise Exception("Unrecognized simulation type. %r" % args.type[0])

    # Run the selected simulation and save to the results to simulations\[<input name>]-[<params>]
    sim.run(iterations=args.iterations, run_name=args.run_name, updates_per_frame=args.updates_per_frame,
            visual=args.visual)

    video_path = glob.glob(os.path.join(sim.output_path, '*.avi'))[0]
    open_file(video_path)