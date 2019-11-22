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
import numpy as np
from scipy import ndimage
from PIL import Image
import os
import logging
import cv2
np.set_printoptions(precision=2)

# Constant Parameters
OUTPUT_FOLDER = "simulations"
DEFAULT_LAPLACE_MATRIX = np.array([[0.05, 0.2, 0.05],
                                   [0.2, -1, 0.2],
                                   [0.05, 0.2, 0.05]], np.float64)

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
    def __init__(self, n=2, orders = [-1, -1], diffusions = [1, 1], feed=0.0545, kill=0.03, length=100, maxConc=3,
                 laplace_matrix=DEFAULT_LAPLACE_MATRIX):
        """
        Object to simulate Gray-Scott diffusion with n particles.
        :param n: number of particles
        :param orders: array of reaction orders for each substrate molecule
                       if the order is -1, the default Gray-Scott equation is used instead of an equation based on reaction kinetics
        :param diffusions: array of diffusion constants for each molecule
        :param feed: feed rate
        :param kill: kill rate
        :param length: side-length of simulation square
        :param maxConc: max concentration
        :param laplace_matrix: 3x3 convolution matrix
        :return None
        """

        # Class parameters
        self.numParticles = n
        self.orders = orders
        self.length = length
        self.maxConc = maxConc
        self.feed = feed
        self.kill = kill
        self.laplace_matrix = laplace_matrix
        self.laplacians = np.zeros((n, length, length))

        # Init particles
        self.particleList = []
        for i in range(n):
            self.particleList.append(Particle(nX=length, nY=length, diffusion=diffusions[i]))

        # Set initial particle state
        self.set_initial_state()

    def set_initial_state(self):
        """
        Sets the initial state of the particle objects for the simulation.
        Currently places 0.5 relative concentration of first 2 input molecules everywhere
        TODO: make this dynamic
        """

        for particle in range(self.numParticles - 1):
            for i in range(self.length):
                for j in range(self.length):
                    self.particleList[particle].blocks[(i, j)] = 0.5

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
        if using == "matplotlib":
            self.run_matplotlib(iterations, run_name)

        else:
            self.run_cv2(iterations, run_name)

        # Change back to the original directory
        os.chdir(cur_dir)

    def run_matplotlib(self, iterations, run_name):
        """
        Use matplotlib as graphics library for simulation run
        :param iterations: Number of iterations for simulation
        :param run_name: Prefix for frame files ("<run_name>-<frame_number>.png")
        :return: None
        """

        for frame in tqdm(range(iterations)):

            binaryArray = np.zeros((self.length, self.length), 'int')

            for i in range(self.length):
                for j in range(self.length):

                    if self.A.getBlock((i, j)) > self.B.getBlock((i, j)):
                        binaryArray[i, j] = 1

            plt.figure()
            plt.imshow(binaryArray)
            plt.savefig(str(run_name) + '-' + str(frame) + '.png')
            plt.close()
            self.update()

    def run_cv2(self, iterations, run_name):
        """
        Use OpenCV as graphics library to create a video of the simulation.
        :param iterations: Number of iterations for simulation
        :param run_name: Prefix for frame files ("<run_name>-<output_type>.png")
        :return: None
        """

        #Start creating video
        simVideo = cv2.VideoCapture(0)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output = cv2.VideoWriter(run_name + "-video.avi", fourcc,8,(self.length,self.length))

        #Create numpy arrays to track total concentrations of all particles and rate or product formation and figures to display concentrations and rate
        particleConcentrations = np.zeros((iterations, self.numParticles, self.length, self.length))
        rate = np.zeros(iterations)
        timesteps = np.arange(iterations)

        for frame in tqdm(range(iterations)): #Run simulation for 1000 timesteps
            #Get concentration for current frame
            for i in range(self.numParticles):
                particleConcentrations[frame,i,:,:] = np.asarray(self.particleList[i].getGrid())

            #If past current frame, get rate of product formation
            if(frame > 0):
                rate[frame] = sum(sum(particleConcentrations[frame,self.numParticles-1,:,:])) - sum(sum(particleConcentrations[frame-1,self.numParticles-1,:,:]))

            #Ensure no values are above max or below min (this shouldn't be an issue if update parameters are set appropriately)
            np.where(particleConcentrations>1, particleConcentrations, 1)
            np.where(particleConcentrations<0, particleConcentrations, 0)

            #Create RGB array
            rgbArray = np.zeros((self.length,self.length,3), 'uint8')
            for i in range(self.numParticles):
                rgbArray[:,:, i] = particleConcentrations[frame,i,:,:]*255.0
            
            #Save as video
            img = Image.fromarray(rgbArray)
            img = np.array(img)
            output.write(img)
            
            self.update()
        output.release()
        cv2.destroyAllWindows()

        #Plot concentration and rate
        colors = ['r', 'g', 'b']
        for i in range(self.numParticles):
            plt.plot(timesteps, sum(sum(particleConcentrations[i,:,:,:])), color=colors[i],label=('particle' + str(i)))
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
            self.laplacians[i,:,:] = ndimage.convolve(np.asarray(self.particleList[i].getGrid()), self.laplace_matrix)


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
        threshold_C = 1/(1+np.exp(-1*conc_C)) if self.numParticles > 2 else 0
        react_C = 0 
        react_AB = 0
        if random.random() <= threshold_C:
            react_C = 1
        if(self.orders[0] == -1):
            dAdt = conc_A + self.diffusions[particle] * lapA - conc_A*conc_B**2 + self.feed * (1 - conc_A)
            dBdt = conc_B + self.B.diffusion * lapB + conc_A*conc_B**2 - (self.kill + self.feed) * conc_B
            dCdt = 0
        elif(self.orders[0] == 1):
            threshold_AB = 1/(1+np.exp(-1*conc_A*conc_B))
            if random.random() <= threshold_AB:
                react_AB = 1
            dAdt = self.particleList[0].diffusion*lapA/9 - react_AB*(min(conc_A, conc_B)) + react_C*conc_C
            dBdt = self.particleList[1].diffusion*lapB/9- react_AB*(min(conc_A, conc_B)) + react_C*conc_C
            dCdt = self.particleList[2].diffusion*lapC/9+ react_AB*(min(conc_A, conc_B)) - react_C*conc_C
        elif(self.orders[0] == 2):
            threshold_AB = 1/(1+np.exp(-1*(conc_A**2)*conc_B))
            if random.random() <= threshold_AB:
                react_AB = 1
            dAdt = self.particleList[0].diffusion*lapA/9 - 2*react_AB*(min(conc_A, conc_B)) + 2*prob_C*conc_C
            dBdt = self.particleList[1].diffusion*lapB/9- react_AB*(min(conc_A, conc_B)) + prob_C*conc_C
            dCdt = self.particleList[2].diffusion*lapC/9 + react_AB*(min(conc_A, conc_B)) - prob_C*conc_C
        return [dAdt, dBdt, dCdt]

    def get_dAdt_at(self, i, j):
        """
        Compute the change in A at a particular location (i, j). Returned equation is
        defined by Gray-Scott model.
        :param i: x location to update
        :param j: y location to update
        :return: dAdt evaluated at (i, j)
        """
        conc_A = self.A.blocks[(i, j)]
        conc_B = self.B.blocks[(i, j)]
        lapA = self.lapA[(i, j)]
        return self.A.diffusion * lapA - conc_A*conc_B**2 + self.feed * (1 - conc_A)

    # TODO: make these dynamic to the particle
    def get_dBdt_at(self, i, j):
        """
        Compute the change in B at a particular location (i, j). Returned equation is
        defined by Gray-Scott model.
        :param i: x location to update
        :param j: y location to update
        :return: dBdt evaluated at (i, j)
        """
        conc_A = self.A.blocks[(i, j)]
        conc_B = self.B.blocks[(i, j)]
        lapB = self.lapB[(i, j)]
        return self.B.diffusion * lapB + conc_A*conc_B**2 - (self.kill + self.feed) * conc_B


if __name__ == "__main__":

    # Run an example simulation and save the results to simulations\simple-[<params>]
    sim = Simulation(n=3, orders=[1,1,1], diffusions = [1, 1, 0.5], feed=0.0545, kill=0.03, length=100)
    sim.run(iterations=100, using="cv2", run_name="simple")