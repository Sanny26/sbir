import cv2
import numpy as np 
import matplotlib.pyplot as plt
import pdb
import math

from operator import itemgetter
from scipy.spatial.distance import cdist


def plot_feature_points(patch, rpoints, patch_centre):
	patch2 = cv2.cvtColor(patch,cv2.COLOR_GRAY2RGB)
	patch2[rpoints[:, 0], rpoints[:, 1], :] = (0, 255, 0)
	patch2[patch_centre[0], patch_centre[1], :] = (255, 0, 0)
	plt.figure()
	plt.subplot(1, 2, 1)
	plt.imshow(patch, cmap='gray')
	plt.subplot(1, 2, 2)	
	plt.imshow(patch2)
	plt.show()


def get_spark_descriptors(img, npoints=500, nbins_r=5, nbins_theta=12):
	if np.count_nonzero(img==255) == 0:
		return 0
	points = (255-img).nonzero()
	rand_ind = np.random.randint(0, len(points[0]), size=(npoints))
	rpoints = [points[0][rand_ind], points[1][rand_ind]]
	rpoints = np.array(rpoints).transpose()
	sketch_points = []
	window_size = 50
	step_size = 5
	half_window_size = int(window_size/2)
	
	bins_r = np.linspace(0, window_size*np.sqrt(2)/2, nbins_r)
	bins_theta = np.linspace(-np.pi, np.pi, nbins_theta)
	descriptor = []
	hist_shape = (len(bins_r), len(bins_theta))
	for i, point in enumerate(rpoints):
		pt_hist = np.zeros(hist_shape)

		xlow = max(0, point[0]-half_window_size)
		xhigh = min(img.shape[0], point[0]+half_window_size)
		ylow = max(0, point[1]-half_window_size)
		yhigh = min(img.shape[1], point[1]+half_window_size)
		img[point[0], point[1]] = 2
		patch = img[xlow: xhigh, ylow:yhigh]
		patch_centre = (patch==2).nonzero()
		patch_centre = (patch_centre[0][0], patch_centre[1][0])
		img[point[0], point[1]] = 0
		if np.count_nonzero(patch==255) == 0:
			continue
		sketch_points = (patch==255).nonzero()
		sketch_points = np.array(sketch_points).transpose()

		## bin directions to get better results.
		if sketch_points.shape[0]>1:
			theta_array = cdist(np.array([patch_centre]), sketch_points, lambda u, v: math.atan2((v[1] - u[1]), (v[0] - u[0])))
			direction_bins = np.arange(theta_array.min(), theta_array.max(), 0.05)
			direction_array = []
			for each in theta_array[0]:
				dbin = (each>=direction_bins).nonzero()[0][-1]
				direction_array.append(direction_bins[dbin])

			## get first feature line points in every direction.
			direction_array = np.array(direction_array)
			unique_directions = np.unique(direction_array)
			rpoints = []
			for direction in unique_directions:
				pos = (direction_array == direction).nonzero()[0]
				dpoints = sketch_points[pos]
				darray = cdist(np.array([patch_centre]), dpoints)
				rpoints.append(sketch_points[pos[darray.argmin()]])
				bin_r = (bins_r <= darray[0, darray.argmin()]).nonzero()[0][-1] 	
				bin_theta = (bins_theta <= direction).nonzero()[0][-1]
				pt_hist[bin_r, bin_theta] += 1 	
				
			rpoints = np.array(rpoints)
			#plot_feature_points(patch, rpoints, patch_centre)
			descriptor.append(pt_hist)

		else:
			bin_r = (bins_r <= darray[0, darray.argmin()]).nonzero()[0][-1] 	
			bin_theta = (bins_theta <= direction).nonzero()[0][-1]
			pt_hist[bin_r, bin_theta] += 1
			descriptor.append(pt_hist)

	return descriptor
		
			
if __name__ == "__main__":
	img = cv2.imread('data/test_edge.png', 0)
	get_spark_descriptors(img)


