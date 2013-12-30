'''
Jason Ting, Alex Lehman, Ryan Lee
CS205 Final Project
Parallelized Implementation of Discretized Poisson Blending Algorithm using CUDA
'''

from PIL import Image
import numpy as np
import time
import glob
import pycuda.autoinit
import pycuda.driver as cu
import pycuda.compiler as nvcc
import pycuda.gpuarray as gpu
from Cheetah.Template import Template

# define the CUDA kernels for the mask and blending
mask_source = """
// define the interior pixels and make border pixels white
__global__ void mask_kernel(uchar3* source, int width, int height)
{
	// Compute thread id in x, y, and coalesced
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    int tid = i * width + j;

    // ensure each pixel is within image size and not white
    if (i >= 0 && i < height && j >= 0 && j < width && (source[tid].x < 255 || source[tid].y < 255 || source[tid].z < 255)) {
    	// set up calculcations
    	int pos;

    	// goes over neighbors (up, down, left, right)
    	for (int j = 0; j < 4; j++) {
		    // define position for neighbor
		    pos = tid + ((j-2) % 2) + ((int)((j-2) / 2)) * width;

		    // changes pixels in the border to white 
		    if (source[pos].x == 255 && source[pos].y == 255 && source[pos].z == 255) {
		    	for (int i = 0; i < 3; i++) {
		    		switch(i) {
		    			case 0:
		    				source[tid].x = 255;
		    				break;
		    			case 1:
		    				source[tid].y = 255;
		    				break;
		    			case 2:
		    				source[tid].z = 255;
		    				break;
		    		}
		    	}
		    }
    	}
    }
}
"""

poisson_blending_source = """
__global__ void poisson_blending_kernel(uchar3* source, uchar3* destination, uchar3* buffer, int width, int height)
{
    // Compute thread id in x, y, and coalesced
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    int tid = i * width + j;
    unsigned char source_pix;
    unsigned char source_pos;
    unsigned char dest_pix;
    unsigned char dest_pos;
    unsigned char buffer_pos;
    float next_buffer_x = 0.0;
    float next_buffer_y = 0.0;
    float next_buffer_z = 0.0;

    // ensure each pixel is within image size and not white
    if (i >= 0 && i < height && j >= 0 && j < width && (buffer[tid].x < 255 || buffer[tid].y < 255 || buffer[tid].z < 255)) {
    	// set up calculations for next buffer
		int pos;
		float sum;

		// iterates over RGB
    	for (int i = 0; i < 3; i++) {
    		switch(i) {
    			case 0:
    				source_pix = source[tid].x;
    				dest_pix = dest[tid].x;
    				break;
    			case 1:
    				source_pix = source[tid].y;
    				dest_pix = dest[tid].y;
    				break;
    			case 2:
    				source_pix = source[tid].z;
    				dest_pix = dest[tid].z;
    				break;
    		}

		    // setup calculations
		    sum = 0.0;

		   	// iterates over neighbors (up, down, left, right)
		    for (int j = 0; j < 4; j++) {
		    	// define position for neighbor
		    	pos = tid + ((j-2) % 2) + ((int)((j-2) / 2)) * width;
		    	switch(i) {
    				case 0:
    					source_pos = source[pos].x;
    					dest_pos = dest[pos].x;
    					buffer_pos = buffer[pos].x;
    					break;
    				case 1:
    					source_pos = source[pos].y;
   						dest_pos = dest[pos].y;
   						buffer_pos = buffer[pos].y;
   						break;
    				case 2:
    					source_pos = source[pos].z;
    					dest_pos = dest[pos].z;
    					buffer_pos = buffer[pos].z;
    					break;
   				}

		    	// adds buffer neighbors if pixel is in interior otherwise add destination neighbors
		    	if (buffer[pos].x < 255 || buffer[pos].y < 255 || buffer[pos].z < 255)
		   			sum += buffer_pos;
		   		else
		   			sum += dest_pos;

		    	//add difference between source and neighbor
		    	sum += (source_pix - source_pos);
		   	}

		   	// updates the next buffer and clip (0,255)
		   	switch(i) {
		   		case 0:
		   			next_buffer_x = min(255.f, max(0.f, sum/4.f));
		   			break;
		   		case 1:
		   			next_buffer_y = min(255.f, max(0.f, sum/4.f));
		   			break;
		   		case 2:
		   			next_buffer_z = min(255.f, max(0.f, sum/4.f));
		   			break;
		   	}
	    }
	}

	// updates the destination image and buffer
	destination[tid] = make_uchar3(next_buffer_x, next_buffer_y, next_buffer_z);
	buffer[tid] = make_uchar3(next_buffer_x, next_buffer_y, next_buffer_z);
	}
}
"""

def cuda_compile(source_string, function_name):
	# compile the CUDA Kernel at runtime
	source_module = nvcc.SourceModule(source_string)
	# return a handle to the compiled CUDA kernel
	return source_module.get_function(function_name)

def interior_buffer(source_im, dest_im, b_size, g_size, RGB, neighbors):
  	# compile the CUDA kernel
  	mask_kernel = cuda_compile(mask_source, "mask_kernel")

  	# alloc memory to GPU
  	d_source = cu.mem_alloc(source_im.nbytes)
  	cu.memcpy_htod(d_source, source_im)

  	# sends to GPU filter out interior points in the mask
  	mask_kernel(d_source, dest_im.shape[1], dest_im.shape[0], block=b_size, grid=g_size)

  	# retrieves interior point buffer from GPU
  	inner_buffer = np.array(dest_im, dtype =np.uint8)
  	cu.memcpy_dtoh(inner_buffer, d_source)

  	# returns the interior buffer
  	return inner_buffer

def poisson_parallel(source_im, dest_im, b_size, g_size, RGB, neighbors, interior_buffer, n):
	# compile the CUDA kernel
  	poisson_blending_kernel = cuda_compile(poisson_blending_source, "poisson_blending_kernel")

  	# alloc memory in GPU
  	out_image = np.array(dest_im, dtype = np.uint8)
  	d_source, d_destination, d_buffer = cu.mem_alloc(source_im.nbytes), cu.mem_alloc(dest_im.nbytes), cu.mem_alloc(interior_buffer.nbytes)
  	cu.memcpy_htod(d_source, source_im)
  	cu.memcpy_htod(d_destination, dest_im)
  	cu.memcpy_htod(d_buffer, interior_buffer)

  	# calls CUDA for Poisson Blending n # of times
  	for i in range(n):
		poisson_blending_kernel(d_source, d_destination, d_buffer, dest_im.shape[1], dest_im.shape[0], block=b_size, grid=g_size)

	# retrieves the final output image and returns
	cu.memcpy_dtoh(out_image, d_destination)
  	return out_image


if __name__ == '__main__':
	# checks for proper usage
	if len(argv) == 2:
		source_files = glob.glob('source*.jpg')
		dest_files = glob.glob('dest*.jpg')
		N = argv[1]
		if len(source_files) != len(dest_files):
			print "Please make sure that your files are named sourceN.jpg, destN.jpg, and that each source is paired with a dest image."
	elif len(argv) == 4:
		source_files = [argv[1]]
		dest_files = [argv[2]]
		N = argv[3]
	else:
		print "Usage: python", argv[0], "[source image] [destination image] [# iterations] OR python", argv[0], "[# iterations] (for entire directory)"
		exit()

	# iterates over the image files
	for i in range(len(source_files)):
		# load in source/dest images and convert to Numpy arrays for blending with uint8
		source_im = np.array(Image.open(in_file), dtype = np.uint8)
		dest_im = np.array(Image.open(out_file_name[i]), dtype = np.uint8)

		# warmup the GPU (no calculations)
		for k in range(100):
			d_source = gpu.to_gpu(source_im)
			d_dest = gpu.to_gpu(dest_im)
			source_im = d_source.get()
			dest_im = d_dest.get()

		# block size (threads per block)   
		b_size = (16,16,1)    
		# grid size (blocks per grid)
		g_size = (int(np.ceil(float(dest_im.shape[1])/b_size[0])), int(np.ceil(float(dest_im.shape[0])/b_size[1])))

		# initialize color pixel locations and neighboring positions [(+-1,0),(0,+-1)] for Cheetah
		RGB = ['x','y','z']
		neighbors = []
		for j in range(-1,2,2):
			neighbors.append((j,0))
			neighbors.append((0,j))

		# apply Poisson blending and time
		start = time.time()
		inner_buffer = interior_buffer(source_im, dest_im, b_size, g_size, RGB, neighbors)
		out_im = poisson_parallel(source_im, dest_im, b_size, g_size, RGB, neighbors, inner_buffer, N)
		end = time.time()
		print 'Parallel Time: ' + str(end - start) + ' seconds'

		# creates output and save the image
		out_im = Image.fromarray(out_im, 'RGB')
		out_im.save('results_' + str(i) + '.png')
		