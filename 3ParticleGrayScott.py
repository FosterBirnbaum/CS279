#This implements a simple version of a Gray-Scott diffusion model

from numpy import zeros
from random import random
from math import floor,ceil
import matplotlib
matplotlib.use('Agg')
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

#Holds information about each particle
class particle(object):
	def __init__(self,
	nX = 25,
	nY = 25,
	diffusion = 1,
	):
		self.nX = nX
		self.nY = nY
		self.blocks = {(x,y):0 for x in range(nX) for y in range(nY)}
		self.diffusion = diffusion

	def setBlock(self,block,val):
	    if block in self.blocks:
	      self.blocks[block] = val

	def getBlock(self,block):
	    if block in self.blocks:
	      return self.blocks[block]
	    return None

	def getGrid(self):
		grid = zeros((self.nX,self.nY))
		for (x,y) in self.blocks:
		  grid[x,y] = self.blocks[(x,y)]
		return grid

	# Returns the correct x index after moving step blocks
	# to the right (to the left for negative step)
	def nextX(self,x,step):
		return (x+step)%self.nX
	# Returns the correct y index after moving step blocks
	# up (down for negative step)
	def nextY(self,y,step):
		return (y+step)%self.nY

#Calculate first derivative for particle and current state
def firstDeriv(part):
	ddx = {xy:0 for xy in part.blocks}
	ddy = {xy:0 for xy in part.blocks}
	for (i,j) in part.blocks:
		curVal = part.getBlock((i,j))
		nextValX = part.getBlock((part.nextX(i,1), j))
		prevValX = part.getBlock((part.nextX(i,-1), j))
		nextValY = part.getBlock((i, part.nextY(j,1)))
		prevValY = part.getBlock((i, part.nextY(j,-1)))
		ddx[(i,j)] = 0.5*(nextValX - curVal) + 0.5*(curVal - prevValX)
		ddy[(i,j)] = 0.5*(nextValY - curVal) + 0.5*(curVal - prevValY)
	return [ddx, ddy]

def laplacian(part, firstDerivs):
	lap = {xy:0 for xy in part.blocks}
	ddx = np.asarray(firstDerivs[0])
	ddy = np.asarray(firstDerivs[1])
	for (i,j) in part.blocks:
		lapx = 0.5*(ddx[(i,j)] - ddx[(part.nextX(i,-1), j)]) + 0.5*(ddx[(part.nextX(i,1), j)] - ddx[(i,j)])
		lapy = 0.5*(ddy[(i,j)] - ddy[(i, part.nextY(j,-1))]) + 0.5*(ddy[(i, part.nextY(j,1))] - ddy[(i,j)])
		lap[(i,j)] = lapx + lapy
	return lap


def update(A, B, C, feed, kill):
	# Iterate through the keys of self.blocks (i.e. the
	# coordinates) to compute the first discrete derivatives
	firstDerivA = firstDeriv(A)
	firstDerivB = firstDeriv(B) 

	# Iterate through the keys of self.blocks to
	# compute the second discrete derivatives
	laplaceMatrix =  np.array([[0.05, 0.2, 0.05], [0.2, -1, 0.2], [0.05, 0.2, 0.05]], np.float64)
	lapA = ndimage.convolve(np.asarray(A.getGrid()), laplaceMatrix)
	lapB = ndimage.convolve(np.asarray(B.getGrid()), laplaceMatrix)
	lapC = ndimage.convolve(np.asarray(C.getGrid()), laplaceMatrix)

  	#Loop through all points in grid and update
	shape = lapA.shape
	for i in range(shape[0]):
		for j in range(shape[1]):
			newBlockA = A.blocks[(i,j)] + A.diffusion*lapA[(i,j)] - (A.blocks[(i,j)]*B.blocks[(i,j)]) + C.blocks[(i,j)]
			newBlockB = B.blocks[(i,j)] + B.diffusion*lapB[(i,j)] - (A.blocks[(i,j)]*B.blocks[(i,j)]) + C.blocks[(i,j)]
			newBlockC = C.diffusion*lapC[(i,j)] + (A.blocks[(i,j)]*B.blocks[(i,j)])
			A.blocks[(i,j)] = newBlockA
			B.blocks[(i,j)] = newBlockB
			C.blocks[(i,j)] = newBlockC


def main():
	length = 100
	maxConc = 3
	A = particle(nX = length, nY = length, diffusion = 1)
	for x in range(46):
		for y in range(100):
			A.blocks[(x,y)] = 1
	B = particle(nX = length, nY = length, diffusion = 1)
	for x in range(55,100):
		for y in range(100):
			B.blocks[(x,y)] = 1
	C = particle(nX = length, nY = length, diffusion = 0.1) 
	count = 0
	for frame in range(100): #Run simulation for 1000 timesteps
		if (frame % 10 == 0): print((frame / 10))
		frameA = np.asarray(A.getGrid())
		frameB = np.asarray(B.getGrid())
		frameC = np.asarray(C.getGrid())
		frameA *= 255.0
		frameB *= 255.0
		frameC *= 255.0
		np.where(frameA>255, 255, frameA)
		np.where(frameB>255, 255, frameB)
		np.where(frameC>255, 255, frameC)
		rgbArray = np.zeros((length,length,3), 'uint8')
		rgbArray[:,:, 0] = frameA
		rgbArray[:,:, 1] = frameB
		rgbArray[:,:, 2] = frameC
		img = Image.fromarray(rgbArray)
		plt.figure()
		plt.imshow(img)
		plt.savefig('3parttest' + os.sep + 'frame ' + str(frame) + '.png')
		plt.close()
		update(A, B, C, .2, .03)


if __name__ == "__main__":
	main()
