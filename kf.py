import numpy as np
import scipy.stats.chisquare as chisquare
class KF:
	def __init__(self, F = None, B = None, H = None, Q = None, R = None, P = None, x0 = None):
		if(F is None or H is None):
			raise ValueError("|--INVALID PARAMETERS--|")

		self.n = F.shape[1]
		self.m = H.shape[1]

        # Next State Function
		self.F = F
        # Measurement Function
		self.H = H
        # External Motion (dotted with mu in prediction step)
		self.B = 0 if B is None else B
        # Identity Matrix
		self.Q = np.eye(self.n) if Q is None else Q
		self.R = np.eye(self.n) if R is None else R
		self.P = np.eye(self.n) if P is None else P
        # Initial State Matrix (Position & Velocity)
		self.x = np.zeros((self.n, 1)) if x0 is None else x0

	def predict(self, u = 0):
		self.x = np.dot(self.F, self.x) + np.dot(self.B, u)
		self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
		return self.x

	def update(self, z):
		y = z - np.dot(self.H, self.x)
		S = self.R + np.dot(self.H, np.dot(self.P, self.H.T))
		K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
		self.x = self.x + np.dot(K, y)
		I = np.eye(self.n)
		self.P = np.dot(np.dot(I - np.dot(K, self.H), self.P), 
			(I - np.dot(K, self.H)).T) + np.dot(np.dot(K, self.R), K.T)

	@staticmethod
	def classify(raw_fixations, sample_duration=4, minimum_duration=50, threshold=1):
		filter_fixation = []
		raw_fixations_np = np.array(raw_fixations)
		times = raw_fixations_np[:,0]
		dt = sample_duration
		sfreq = 1 / np.mean(times[1:] - times[:-1])
		sample_thresh = sfreq * threshold / 1000
		
		# calculate movement velocities
		x_coord = raw_fixations_np[:,1]
		y_coord = raw_fixations_np[:,2]
		coord = np.stack([x_coord, y_coord])
		
		F = np.array([[1, dt, 0], [0, 1, dt], [0, 0, 1]])
		H = np.array([1, 0, 0]).reshape(1, 3)
		Q = np.array([[0.05, 0.05, 0.0], [0.05, 0.05, 0.0], [0.0, 0.0, 0.0]])
		R = np.array([0.5]).reshape(1, 1)

		for time_frame in range(0,math.ceil(len(x_coord)/window_size)):
			timestamp_now = timestamp[time_frame*window_size:(time_frame+1)*window_size-1]
			x_coord_now = x_coord[(time_frame*window_size):((time_frame+1)*window_size-1)]
			y_coord_now = y_coord[time_frame*window_size:(time_frame+1)*window_size-1]
			remove_coordinates = np.any(np.vstack((x_coord_now <= 0, y_coord_now <= 0, x_coord_now >= 1920, y_coord_now >= 1080)), axis=0)
			timestamp_now = timestamp_now[np.logical_not(remove_coordinates)]
			x_coord_now = x_coord_now[np.logical_not(remove_coordinates)]
			y_coord_now = y_coord_now[np.logical_not(remove_coordinates)]
			coord_now = np.vstack((x_coord_now, y_coord_now))
			if coord_now.shape[1] < 5:
				continue

			sfreq = 1 / np.mean(timestamp_now[1:] - timestamp_now[:-1])
			vels_now = np.linalg.norm(coord_now[:, 1:] - coord_now[:, :-1], axis=0)
			vels_now = np.concatenate([[0.], vels_now])
			vels_now_pred = []
			kf = KF(F = F, H = H, Q = Q, R = R)

			for z in vels_now:
				vels_now_pred.append(np.dot(H, kf.predict())[0])
				kf.update(z)
				chi_square_now,_ = chisquare(vels_now, vels_now_pred)
				if chi_square_now < threshold:
					# a fixation:
					filter_fixation.append[timestamp_now[-1], len(x_coord_now)*4, np.mean(x_coord_now), np.mean(y_coord_now)]
		return filter_fixation
