Parallelization of Poisson Blending
===================================
By Jason Ting, Ryan Lee, Alex Lehman

Final Project for Computer Science 205, Fall 2013 (Cris Cecka)
--------------------------------------------------------------

The objective of the Poisson Blending algorithm is to compose a source image and a target image in the gradient domain. The code implements Poisson Blending in parallel with CUDA and Cheetah to efficiently and automatically superimpose images without visible seams.

How to Run:
-----------
There are two ways to run the code:

1) Using the images included in the folder and the course software load, execute the following on the Resonance node:  
$ python parallel_poisson.py [# iterations]

2) Specifying the image that you would like to process, execute the following, again on the Resonance node:  
$ python parallel_poisson.py [source image] [destination image] [# iterations]

Benchmarking:
-------------
For the purposes of analysis, the average time per iteration was computed over 800 iterations (N) for destination images of 5 sizes: (200, 142), (375, 266), (750, 531), (1500, 1062), and (2500, 1770).
