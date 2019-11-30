"""
This file defines the particle and simulation classes that allow users
to run a variety of simulation types. The available types are as follows:
    -TwoParticleGS: Simulates normal Gray-Scott (2B + A --> 3B, feed rate for A, kill rate for B)
    -ThreeParticleFirstOrder: Simulates a reaction involving 3 particles (A + B --> C) that is first order in A and B
    -ThreeParticleSecondOrder: Simulates a reaction involving 3 particles (A + B --> C) that is second order in A and first order in B
    -ThreeParticleFirstOrderWithCatalyst: Simulates a reaction involving 3 particles (A + B --> C) that is first order in substrates and occurs in presence of catalyst
    -Custom: Allows user to define input parameters
All simulations are saved in video format, as are plots of relative concentrations of all species involved and a plot of the rate of formation of product (if one is involved).

Written by Collin Schlager and Foster Birnbaum (fosb@stanford.edu) -- 12/6/2019
The backbone of the particle class was written by instructors of CS279 in Fall 2019
"""

from numpy import zeros
from numpy import random
from tqdm import tqdm
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter
import numpy as np
from scipy import ndimage
import scipy.stats as stats
import os
import sys
import logging
from multiprocessing import Pool
import time


np.set_printoptions(precision=2)

# Constant Parameters
OUTPUT_FOLDER = "simulations"
DEFAULT_INIT = "random"
DEFAULT_LAPLACE_MATRIX = np.array([[0.05, 0.2, 0.05],
                                   [0.2, -1.0, 0.2],
                                   [0.05, 0.2, 0.05]], np.float64)
DEFAULT_TEMP_LOWERBOUND = 273
DEFAULT_TEMP_UPPERBOUND = 473

"""
Defines the particle class that contains information on a specific particle in the simulation.
Each particle is defined as an nX by nY grid, with values in each grid box representing the 
concentration of that particle at that position.
"""
class Particle(object):
    def __init__(self, nX=25, nY=25, diffusion=1.0):
        self.nX = nX
        self.nY = nY
        self.blocks = {(x, y): 0 for x in range(nX) for y in range(nY)}
        self.diffusion = diffusion

    def getBlock(self, block):
        if block in self.blocks:
            return self.blocks[block]
        else:
            raise

    def getGrid(self):
        grid = zeros((self.nX, self.nY))
        for (x, y) in self.blocks:
            grid[x, y] = self.blocks[(x, y)]
        return grid

"""
Defines the Simulation object that runs a reaction-diffusion simulation according
to user input parameters. For a list of sample input parameters, please see top of file comment.
"""
class Simulation(object):
    def __init__(self, n=2, orders = [-1, -1], diffusions = [1, 1], feed=0.0545, kill=0.03, activationEnergies = [1, 1], temp = 298, 
                length=100, maxConc=3, laplace_matrix=DEFAULT_LAPLACE_MATRIX, init = DEFAULT_INIT):
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
        :param intit: string identifying how grid shoud be initialized
        :return None
        """

        # Class parameters
        self.numParticles = n
        self.orders = orders
        self.length = length
        self.maxConc = maxConc
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
        #If user wants random intialization, put first two particles to 0.5 concentration everywhere
        if self.init == "random":
            for particle in range(self.numParticles - 1):
                for i in range(self.length):
                    for j in range(self.length):
                        self.particleList[particle].blocks[(i, j)] = 0.5
        #If user wants point mass, put first particle to 1 everywhere and second particle to 0 everywhere except a small region in center
        elif self.init == "pointMass":
            for i in range(self.length):
                for j in range(self.length):
                    self.particleList[0].blocks[(i, j)] = 1
                    if (self.length*0.5 - i)**2 + (self.length*0.5 - j)**2 <= self.length*0.05:
                        self.particleList[1].blocks[(i,j)] = 1
                        self.particleList[0].blocks[(i, j)] = 0
                    else:
                        self.particleList[1].blocks[(i,j)] = 0


    def run(self, iterations, using="cv2", run_name="simple"):
        """
        Runs the simulation and stores frames to a result directory in `simulatons/`
        :param iterations: number of iterations to run the simulation for
        :param using: which graphics library to use ["matplotlib" or "cvs"]. Default is "cv2"
        :param run_name: name of the simulation run. Used as prefix for output directory
        :return: None
        """

        # Create and go to output directory
        output_dir_name = "{}_iterations-{}_length-{}_feed-{}_kill-{}".format(run_name,
                                                                              iterations,
                                                                              self.length,
                                                                              self.feed,
                                                                              self.kill)

        try:
            output_path = os.path.join(OUTPUT_FOLDER, output_dir_name)
            cur_dir = os.getcwd()
            os.mkdir(output_path)
            os.chdir(output_path)
        except FileExistsError as e:
            logging.error("Simulation already exists with same name and parameters. "
                            + "Please change run_name or delete the existing directory.")
            return

        # Run the proper simulation
        self.run_matplotlib(iterations, run_name)

        # Change back to the original directory
        os.chdir(cur_dir)


    def run_matplotlib(self, iterations, run_name):
        """
        Use matplotlib as graphics library to create a video of the simulation.
        :param iterations: Number of iterations for simulation
        :param run_name: Prefix for frame files ("<run_name>-<output_type>.png")
        :return: None
        """

        #Start creating video
        metadata = dict(title='Movie Test', artist='Matplotlib', comment='Movie support!')
        writer = FFMpegWriter(fps=15, metadata=metadata)
        fig = plt.figure()
        ims = []

        #Create numpy arrays to track total concentrations of all particles and rate or product formation and figures to display concentrations and rate
        particleConcentrations = np.zeros((iterations, self.numParticles, self.length, self.length), np.float64)
        rate = np.zeros(iterations, np.float64)
        timesteps = np.arange(iterations)
        
        for frame in tqdm(range(iterations)):
            #Get concentration for current frame
            for i in range(self.numParticles):
                particleConcentrations[frame,i,:,:] = np.asarray(self.particleList[i].getGrid())

            #If past current frame, get rate of product formation
            if(frame > 0):
                rate[frame] = sum(sum(particleConcentrations[frame,self.numParticles-1,:,:])) - sum(sum(particleConcentrations[frame-1,self.numParticles-1,:,:]))

            #Ensure no values are above max or below min (this shouldn't be an issue if update parameters are set appropriately)
            np.where(particleConcentrations<1, particleConcentrations, 1)
            np.where(particleConcentrations>0, particleConcentrations, 0)

            im = plt.imshow(particleConcentrations[frame, 0, :, :], animated=True)
            ims.append([im])

            #Save individual frames (default commented out)
            """
            plt.figure()
            plt.imshow(particleConcentrations[frame, 0, :, :])
            plt.savefig(str(run_name) + '-' + str(frame) + '.png')
            plt.close()
            """
            self.update()
            #for step in range(25):
            #    self.update()
        
        #Save video
        ani = animation.ArtistAnimation(fig, ims, interval=10, blit=True, repeat_delay=1000)
        ani.save(run_name + "-video.avi")
        plt.close()

        #Plot concentration and rate
        colors = ['r', 'g', 'b']
        for i in range(self.numParticles):
            plt.plot(timesteps, np.sum(np.sum(particleConcentrations[:,i,:,:],axis=-1),axis=-1)/(self.length**2), color=colors[i],label=('particle' + str(i)))
        plt.legend()
        plt.savefig(str(run_name) + '-concentrations.png')
        plt.close()

        plt.plot(timesteps, rate, color=colors[0],label=('rate'))
        plt.legend()
        plt.savefig(str(run_name) + '-rate.png')
        plt.close()

    def update(self):
        """
        Update step. Updates blocks of all types of particles using the framework input by user.
        :return: None
        """

        # Update the laplacians for all particles for this time step
        self.compute_laplacians()

        # Loop through all points in grid and update using d[Particle]/dt as defined by Gray-Scott or reaction kinetics
        for i in range(self.length):
            for j in range(self.length):
                concentrationUpdates = self.get_dParticlesdt_at(i, j)
                for particle in range(self.numParticles):
                    self.particleList[particle].blocks[(i, j)] += concentrationUpdates[particle]

    def compute_laplacians(self):
        """
        Compute the laplacians for all particles using the 3x3 convolution matrix.
        :return: None
        """
        for i in range(self.numParticles):
            self.laplacians[i,:,:] = ndimage.convolve(np.asarray(self.particleList[i].getGrid()), self.laplace_matrix, mode='wrap')


    def compute_maxwell(self):
        """
        Compute the energy of the system according to the maxwell-boltzmann distribution
        :return: energy sampled from appropriate maxwell-boltzmann distribution
        """
        maxwell = stats.maxwell
        scale = 10*((self.temp - DEFAULT_TEMP_LOWERBOUND)/(DEFAULT_TEMP_UPPERBOUND - DEFAULT_TEMP_LOWERBOUND))
        energy = maxwell.rvs(loc=0, scale=scale, size=1)
        return energy[0]

    def get_dParticlesdt_at(self, i, j):
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
        #TODO: Make this more dynamic (allow orders of second particle to matter, too)
        conc_A = self.particleList[0].blocks[(i, j)]
        conc_B = self.particleList[1].blocks[(i, j)]
        conc_C = self.particleList[2].blocks[(i, j)] if self.numParticles > 2 else 0
        lapA = self.laplacians[0, i, j]
        lapB = self.laplacians[1, i, j]
        lapC = self.laplacians[2, i, j] if self.numParticles > 2 else 0
        react_AB = 0
        react_C = 0
        if(self.orders[0] == -1):
            dAdt = self.particleList[0].diffusion*lapA - conc_A*(conc_B**2) + self.feed*(1 - conc_A)
            dBdt = self.particleList[1].diffusion*lapB + conc_A*(conc_B**2) - (self.kill+self.feed)*conc_B
            dCdt = 0
        elif(self.orders[0] == 1):
            curEnergy = self.compute_maxwell()
            if(curEnergy > self.activationEnergies[0]):
                react_AB = 1
            if(curEnergy > self.activationEnergies[1]):
                react_C = 1
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
            threshold_AB = 1/(1+np.exp(-1*(conc_A**2)*conc_B))
            if random.random() <= threshold_AB:
                react_AB = 1
            dAdt = self.particleList[0].diffusion*lapA - react_AB*(conc_A*conc_B**2) + react_C*conc_C
            dBdt = self.particleList[1].diffusion*lapB - react_AB*(conc_A*conc_B**2) + react_C*conc_C
            dCdt = self.particleList[2].diffusion*lapC + react_AB*(conc_A*conc_B**2) - react_C*conc_C
        return [dAdt, dBdt, dCdt]

if __name__ == "__main__":
    #Decide what type of simulation the user wants to run
    if sys.argv[1] == "2Particles":
        sim = Simulation(n=2, orders=[-1,-1], diffusions = [1, 0.5], feed = 0.0362, kill = 0.062, length=50, init="pointMass")
    elif sys.argv[1] == "2ParticlesZeroOrder":
        sim = Simulation(n=2, orders=[0, 0], diffusions = [1, 0.5], activationEnergies = [3, 3], temp = 298, length=50, init="pointMass")
    elif sys.argv[1] == "3ParticlesFirstOrder":
        sim = Simulation(n=3, orders=[1,1,1], diffusions = [1, 1, 0.5], feed=0.032, kill=0.0062, length=50, init="random")
    
    #Run the selected simulation and save to the results to simulations\[<input name>]-[<params>]
    sim.run(iterations=750, using="matplotlib", run_name=sys.argv[2])