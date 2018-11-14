import cv2
import numpy as np 
import matplotlib.pyplot as plt
import pdb
from matplotlib import path as Path
import matplotlib.patches as patches


def get_spark_descriptors(img, npoints=500):
	points = (1-img).nonzero()
	rand_ind = np.random.randint(0, len(points[0]), size=(npoints))
	rpoints = [points[0][rand_ind], points[1][rand_ind]]
	rpoints = np.array(rpoints).transpose()
	sketch_points = []
	window_size = 50
	step_size = 5
	half_window_size = int(window_size/2)
	for point in rpoints:
		xlow = max(0, point[0]-half_window_size)
		xhigh = min(img.shape[0], point[0]+half_window_size)
		ylow = max(0, point[1]-half_window_size)
		yhigh = min(img.shape[1], point[1]+half_window_size)
		img[point[0], point[1]] = 2
		'''nimg = img.copy()
		nimg[point[0]-5: point[0]+5, point[1]-5:point[1]+5, :] = (0, 255, 0)
		print(point)'''
		patch = img[xlow: xhigh, ylow:yhigh]
		patch_centre = (patch==2).nonzero()
		patch_centre = (patch_centre[0][0], patch_centre[1][0])
		img[point[0], point[1]] = 1
		if np.count_nonzero(patch)==0:
			continue
		start = 0
		for step in range(start, patch.shape[0], step_size):
			sub_patch = Path([patch_centre, (0, start), (0, step), patch_centre])
			cv2.line(patch, patch_centre, (0, start), color=1)
			cv2.line(patch, (0, start), (0, step), color=1)
			cv2.line(patch, patch_centre, (0, step), color=1)
			plt.imshow(patch, cmap='gray')
			plt.show()
			pdb.set_trace()
			
			
			
if __name__ == "__main__":
	img = cv2.imread('data/test_edge.png', 0)
	get_spark_descriptors(img)


