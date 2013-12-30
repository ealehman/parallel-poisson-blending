'''
Jason Ting, Alex Lehman, Ryan Lee
CS205 Final Project
Serial Implementation of Discretized Poisson Blending Algorithm
'''

import numpy as np
import os
from sys import argv
import Image
import time

def mask(source_im):
	# Compute list of tuples of mask from source image
	mask = []
	for i in range(source_im.shape[0]):
		for j in range(source_im.shape[1]):
			if np.all(source_im[i,j] != [255, 255, 255]):
				mask.append((i,j))

	# Compute border and interior tuples
	interior = []
	for i,j in mask:
		if ((i,j+1) in mask) and ((i,j-1) in mask) and ((i+1,j) in mask) and ((i-1,j) in mask):
			interior.append((i,j))
	return interior

def poisson_serial(source_im, dest_im, out_im, interior, buffer1, buffer2, N):

	# Do Jacobi iterations (800 times)
	for color in [0,1,2]:

		# Initialize first buffer to incoming source image
		buffer1 = source_im[:,:,color]

		# Compute each iteration
		for count in range(N):
			print count 

			for i,j in interior:

				# Compute sum 1 and 2
				sum1 = 0
				sum2 = 0

				for k,l in [(i,j+1), (i,j-1), (i+1,j), (i-1,j)]:
					if (k,l) in interior:
						sum1 += buffer1[k,l]
					else:
						sum1 += dest_im[k,l,color]

					sum2 += (source_im[i,j,color] - source_im[k,l,color])

				buffer2[i,j] = min(255, max(0, (sum1 + sum2) / float(4)))

			# Set buffer 1 to buffer 2 and iterate
			buffer1 = buffer2

		# Copy to an output image
		for i,j in interior:
			out_im[i,j,color] = buffer1[i,j]

	return out_im

if __name__ == '__main__':
	if len(argv) != 4:
		print "Usage: python", argv[0], "[source image] [destination image] [number of iterations]"
		exit()

	# Number of iterations
	N = int(argv[3])

  # Load in source/dest images; convert to Numpy arrays for blending; use uint8 for CUDA as we did in hw5
	source_im = np.array(Image.open(argv[1]), dtype = float)
	dest_im = np.array(Image.open(argv[2]), dtype = float)

	# Allocate buffers the size of dest (optimization: maybe only keep size of source + 1 around so buffer is smaller, then add into image later)
	buffer1 = np.zeros((dest_im.shape[0], dest_im.shape[1]), dtype = float)
	buffer2 = np.zeros((dest_im.shape[0], dest_im.shape[1]), dtype = float)
	out_im = dest_im

	start = time.time()
	interior = mask(source_im)
	out_im = np.uint8(poisson_serial(source_im, dest_im, out_im, interior, buffer1, buffer2, N))
	stop = time.time()

	print str(N) + ' Iterations, Serial Time: ' + str(stop - start) + ' seconds'

	out_im = Image.fromarray(out_im, 'RGB')
	out_im.save('result.png')
	