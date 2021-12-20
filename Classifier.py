import numpy as np
from scipy.spatial import distance_matrix
import pandas as pd
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageEnhance, ImageFont

class IMST_classifier:
	def minimum_spanning_tree(X, copy_X=True):
		"""X are edge weights of fully connected graph"""
		if copy_X:
			X = X.copy()

		if X.shape[0] != X.shape[1]:
			raise ValueError("X needs to be square matrix of edge weights")
		n_vertices = X.shape[0]
		spanning_edges = []

		# initialize with node 0:
		visited_vertices = [0]
		num_visited = 1
		# exclude self connections:
		diag_indices = np.arange(n_vertices)
		X[diag_indices, diag_indices] = np.inf

		while num_visited != n_vertices:
			new_edge = np.argmin(X[visited_vertices], axis=None)
			# 2d encoding of new_edge from flat, get correct indices
			new_edge = divmod(new_edge, n_vertices)
			new_edge = [visited_vertices[new_edge[0]], new_edge[1]]
			# add edge to tree
			spanning_edges.append(new_edge)
			visited_vertices.append(new_edge[1])
			# remove all edges inside current tree
			X[visited_vertices, new_edge[1]] = np.inf
			X[new_edge[1], visited_vertices] = np.inf
			num_visited += 1
		return np.vstack(spanning_edges)

	def imst_classifier(raw_fixations, sample_duration=4, minimum_duration=50,distance_threshold=10):
		"""I-MST classifier based on page 10 of the following benchmarking paper:
			https://digital.library.txstate.edu/bitstream/handle/10877/2577/fulltext.pdf?sequence=1&isAllowed=y
			Notes:
				remember that some data is MSG for mouse clicks.
				some records are invalid with value -1.
				read right eye data only.

		Parameters
		----------
		raw_fixations : list
			a list of fixations information containing timestamp, x_cord, and y_cord

		minimum_duration : int, optional
			minimum duration for a fixation in milliseconds, less than minimum is considered noise.
			set to 50 milliseconds by default

		sample_duration : int, optional
			Sample duration in milliseconds, this is 4 milliseconds based on this eye tracker

		maximum_dispersion : int, optional
			maximum distance from a group of samples to be considered a single fixation.
			Set to 25 pixels by default

		Returns
		-------
		list
			a list where each element is a list of timestamp, duration, x_cord, and y_cord
		"""
		window_size = int(math.ceil(minimum_duration / sample_duration))

		filter_fixation = []

		# Go over each window:
		timestamp = np.array(copy.copy(raw_fixations[1]))
		x_cord = np.array(copy.copy(raw_fixations[2]))
		y_cord = np.array(copy.copy(raw_fixations[3]))

		# Filter (skip) coordinates outside of the screen 1920×1080 px
		remove_coordinates = np.any((x_coord < 0,y_coord < 0,x_coord > 1920,y_coord > 1080))
		timestamp = timestamp[remove_coordinates]
		x_coord = x_coord[remove_coordinates]
		y_coord = y_coord[remove_coordinates]

		# now loop through each time window:
		for time_frame in range(0, window_size):
			timestamp_now = timestamp[time_frame* \
				sample_duration:((time_frame+1)*sample_duration)-1]
			x_coord_now = x_coord[time_frame* \
				sample_duration:((time_frame+1)*sample_duration)-1]
			y_coord_now = y_coord[time_frame* \
				sample_duration:((time_frame+1)*sample_duration)-1]
			coord = np.hstack((x_coord, y_coord))
			# Pairwise matrix using Euclidean distance
			coord_pairwise = distance_matrix(coord, coord)

			# construct MST using the just calculated pairwise matrix and Prim's algorithm
			edge_list = minimum_spanning_tree(coord_pairwise)
			edge_list = np.array(edge_list).T.tolist()
			corresponding_dist = coord_pairwise[tuple(edge_list)]
			# on the matrix, get the entries that is under the threshold:
			fixation_which = np.where(corresponding_dist < distance_threshold)
			fixation_which = np.sort(np.unique(edge_list[fixation_which, ].flatten()))
			filter_fixation.append([timestamp[fixation_which], 4*len(fixation_which), x_coord_now[fixation_which],y_coord_now[fixation_which]])
		return filter_fixation
