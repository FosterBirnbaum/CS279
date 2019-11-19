# This implements a simple version of a Gray-Scott diffusion model

from numpy import zeros
from tqdm import tqdm
from random import random
from math import floor, ceil
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from numpy import array
from scipy import ndimage
from matplotlib import colors
from PIL import Image
import sys
import os
np.set_printoptions(precision=2)


DEFAULT_LAPLACE_MATRIX = np.array([[0.05, 0.2, 0.05],
                                   [0.2, -1, 0.2],
                                   [0.05, 0.2, 0.05]], np.float64)


# Holds information about each particle
class Particle(object):
    def __init__(self, nX=25, nY=25, diffusion=1.0):
        self.nX = nX
        self.nY = nY
        self.blocks = {(x, y): 0 for x in range(nX) for y in range(nY)}
        self.diffusion = diffusion

    def setBlock(self, block, val):
        if block in self.blocks:
            self.blocks[block] = val

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

    # Returns the correct x index after moving step blocks
    # to the right (to the left for negative step)
    def nextX(self, x, step):
        return (x + step) % self.nX

    # Returns the correct y index after moving step blocks
    # up (down for negative step)
    def nextY(self, y, step):
        return (y + step) % self.nY


class Simulation(object):
    def __init__(self, feed=0.0545, kill=0.03, length=100, maxConc=3, name="simple",
                 laplace_matrix=DEFAULT_LAPLACE_MATRIX):

        self.name = name
        self.length = length
        self.maxConc = maxConc
        self.feed = feed
        self.kill = kill
        self.laplace_matrix = laplace_matrix
        self.lapA = None
        self.lapB = None

        self.A = Particle(nX=length, nY=length, diffusion=1)
        self.B = Particle(nX=length, nY=length, diffusion=0.5)

        self.set_initial_state()

    def set_initial_state(self):

        for x in self.A.blocks:
            self.A.blocks[x] = 1

        for x in range(45, 55):
            for y in range(45, 55):
                self.B.blocks[(x, y)] = 1

    def run(self, iterations, using="matplotlib"):

        if using == "matplotlib":
            self.run_matplotlib(iterations)

        else:
            self.run_img(iterations)

    def run_matplotlib(self, iterations):

        for frame in tqdm(range(iterations)):

            binaryArray = np.zeros((self.length, self.length), 'int')

            for i in range(self.length):
                for j in range(self.length):

                    if self.A.getBlock((i, j)) > self.B.getBlock((i, j)):
                        binaryArray[i, j] = 1

            plt.figure()
            plt.imshow(binaryArray)
            plt.savefig(str(self.name) + '-' + str(frame) + '.png')
            plt.close()
            self.update()

    def run_img(self, iterations):

        for frame in tqdm(range(iterations)):

            frameA = np.asarray(self.A.getGrid())
            frameB = np.asarray(self.B.getGrid())

            frameA *= 256.0/self.maxConc
            frameB *= 256.0/self.maxConc
            np.where(frameA > 256, 256, frameA)
            np.where(frameB > 256, 256, frameB)

            rgbArray = np.zeros((self.length, self.length, 3), 'uint8')
            rgbArray[..., 0] = frameA
            rgbArray[..., 2] = frameB
            img = Image.fromarray(rgbArray)
            img.save(str(self.name) + '-' + str(frame) + '.jpeg')

            self.update()

    def update(self):

        # Iterate through the keys of self.blocks to compute the second discrete derivatives
        self.compute_laplacians()

        # Loop through all points in grid and update
        shape = self.lapA.shape
        for i in range(shape[0]):
            for j in range(shape[1]):

                dAdt = self.get_dAdt_at(i, j)
                dBdt = self.get_dBdt_at(i, j)

                self.A.blocks[(i, j)] += dAdt
                self.B.blocks[(i, j)] += dBdt

    def compute_laplacians(self):

        self.lapA = ndimage.convolve(np.asarray(self.A.getGrid()), self.laplace_matrix)
        self.lapB = ndimage.convolve(np.asarray(self.B.getGrid()), self.laplace_matrix)

    def get_dAdt_at(self, i, j):
        conc_A = self.A.blocks[(i, j)]
        conc_B = self.B.blocks[(i, j)]
        lapA = self.lapA[(i, j)]
        return self.A.diffusion * lapA - (conc_A*conc_B)**2 + self.feed * (1 - conc_A)

    def get_dBdt_at(self, i, j):
        conc_A = self.A.blocks[(i, j)]
        conc_B = self.B.blocks[(i, j)]
        lapB = self.lapB[(i, j)]
        return self.B.diffusion * lapB + (conc_A*conc_B)**2 - (self.kill + self.feed) * conc_B


if __name__ == "__main__":

    sim = Simulation(feed=0.0545, kill=0.03, length=100, name="simple")
    sim.run(iterations=100, using="matplotlib")
