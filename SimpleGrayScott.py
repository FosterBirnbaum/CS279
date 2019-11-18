#This implements a simple version of a Gray-Scott diffusion model

from numpy import zeros
from random import random
from math import floor,ceil
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


def update(A, B, feed, kill):
	# Iterate through the keys of self.blocks (i.e. the
	# coordinates) to compute the first discrete derivatives
	firstDerivA = firstDeriv(A)
	firstDerivB = firstDeriv(B) 

	# Iterate through the keys of self.blocks to
	# compute the second discrete derivatives
	laplaceMatrix =  np.array([[0.05, 0.2, 0.05], [0.2, -1, 0.2], [0.05, 0.2, 0.05]], np.float64)
	lapA = ndimage.convolve(np.asarray(A.getGrid()), laplaceMatrix)
	lapB = ndimage.convolve(np.asarray(B.getGrid()), laplaceMatrix)

  	#Loop through all points in grid and update
	shape = lapA.shape
	for i in range(shape[0]):
		for j in range(shape[1]):
			newBlockA = (A.blocks[(i,j)] + A.diffusion*lapA[(i,j)] - (A.blocks[(i,j)]*B.blocks[(i,j)]**2) + feed*(1-A.blocks[(i,j)]))
			B.blocks[(i,j)] = (B.blocks[(i,j)] + B.diffusion*lapB[(i,j)] + (A.blocks[(i,j)]*B.blocks[(i,j)]**2) - (kill+feed)*B.blocks[(i,j)])
			A.blocks[(i,j)] = newBlockA
    	
def main():
	length = 100
	maxConc = 3
	A = particle(nX = length, nY = length, diffusion = 1)
	for x in A.blocks:
		A.blocks[x] = 1
	B = particle(nX = length, nY = length, diffusion = 0.5)
	for x in range(45,55):
		for y in range(45,55):
			B.blocks[(x,y)] = 1
	C = particle(nX = length, nY = length, diffusion = 0.5) 
	for frame in range(100): #Run simulation for 1000 timesteps
		if (frame % 10 == 0): print((frame / 10))
		frameA  = np.asarray(A.getGrid())
		frameB = np.asarray(B.getGrid())
		frameC = np.asarray(C.getGrid())
		binaryArray = np.zeros((length,length), 'int')
		for i in range(length):
			for j in range(length):
				if(A.getBlock((i,j)) > B.getBlock((i,j))): binaryArray[i,j] = 1
		plt.figure()
		plt.imshow(binaryArray)
		plt.savefig('test ' + str(frame) + '.png')
		plt.close()
		#frameA *= 256.0/maxConc
		#frameB *= 256.0/maxConc
		#np.where(frameA>256, 256, frameA)
		#np.where(frameB>256, 256, frameB)
		#rgbArray = np.zeros((length,length,3), 'uint8')
		#rgbArray[..., 0] = frameA
		#rgbArray[..., 2] = frameB
		#img = Image.fromarray(rgbArray)
		#img.save('frame ' + str(i) + '.jpeg')
		update(A, B, .0545, .03)




if __name__ == "__main__":
	main()
