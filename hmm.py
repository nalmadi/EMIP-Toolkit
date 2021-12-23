import numpy as np

def forward(V, a, b, initial_distribution):
	'''
	carries out the long computations in HMM models
	'''
	alpha = np.zeros((V.shape[0], a.shape[0]))
	alpha[0, :] = initial_distribution * b[:, V[0]]

	for t in range(1, V.shape[0]):
		# matrix computation
		for j in range(a.shape[0]):
			alpha[t, j] = alpha[t - 1].dot(a[:, j]) * b[j, V[t]]

	return alpha


def backward(V, a, b):
	'''
	
	'''
	beta = np.zeros((V.shape[0], a.shape[0]))

	# set beta(T) = 1
	beta[V.shape[0] - 1] = np.ones((a.shape[0]))

	# Loop in backward way from T-1 to 1
	# python indexing so loop will be T-2 to 0
	for t in range(V.shape[0] - 2, -1, -1):
		for j in range(a.shape[0]):
			beta[t, j] = (beta[t + 1] * b[:, V[t + 1]]).dot(a[j, :])

	return beta


def baum_welch(V, a, b, initial_distribution, n_iter=4):
	'''
	finds unknown parameters of a hidden Markov model
	'''
	M = a.shape[0]
	T = len(V)

	for n in range(n_iter):
		alpha = forward(V, a, b, initial_distribution)
		beta = backward(V, a, b)

		xi = np.zeros((M, M, T - 1))
		for t in range(T - 1):
			denominator = np.dot(np.dot(alpha[t, :].T, a) * b[:, V[t + 1]].T, beta[t + 1, :])
			for i in range(M):
				numerator = alpha[t, i] * a[i, :] * b[:, V[t + 1]].T * beta[t + 1, :].T
				xi[i, :, t] = numerator / denominator

		gamma = np.sum(xi, axis=1)
		a = np.sum(xi, 2) / np.sum(gamma, axis=1).reshape((-1, 1))

		# Add additional T'th element in gamma
		gamma = np.hstack((gamma, np.sum(xi[:, :, T - 2], axis=0).reshape((-1, 1))))

		K = b.shape[1]
		denominator = np.sum(gamma, axis=1)
		for l in range(K):
			b[:, l] = np.sum(gamma[:, V == l], axis=1)

		b = np.divide(b, denominator.reshape((-1, 1)))

	return (a, b)


def viterbi(V, a, b, initial_distribution):
	'''
	computes the the most likely state sequence in a Hidden Markov model
	'''
	T = V.shape[0]
	M = a.shape[0]

	omega = np.zeros((T, M))
	omega[0, :] = np.log(initial_distribution * b[:, V[0]])

	prev = np.zeros((T - 1, M))

	for t in range(1, T):
		for j in range(M):
			# Same as Forward Probability
			probability = omega[t - 1] + np.log(a[:, j]) + np.log(b[j, V[t]])

			# This is our most probable state given previous state at time t (1)
			prev[t - 1, j] = np.argmax(probability)

			# This is the probability of the most probable state (2)
			omega[t, j] = np.max(probability)

	# Path Array
	S = np.zeros(T)

	# Find the most probable last hidden state
	last_state = np.argmax(omega[T - 1, :])

	S[0] = last_state

	backtrack_index = 1
	for i in range(T - 2, -1, -1):
		S[backtrack_index] = prev[i, int(last_state)]
		last_state = prev[i, int(last_state)]
		backtrack_index += 1

	# Flip the path array since we were backtracking
	S = np.flip(S, axis=0)

	# Convert numeric values to actual hidden states
	result = []
	for s in S:
		if s == 0:
			result.append("A")
		else:
			result.append("B")

	return result


def hmm_classifier(raw_fixations, velocity_threshold=50, sample_duration=4, maximum_dispersion=25):

	'''Hidden Markov Model Identification algorithm from Salvucci & Goldberg (2000)
	
	Input: array of eye position points, velocity threshold, initial,
	transitional, observation probabilities
	
	Output: array of fixations and saccades
	'''

	# Calculate point­to­point velocities for each point in the eye position array
	timestamp=raw_fixations['timestamp']
	sample_freq = 1 / np.mean(timestamp[1:] - timestamp[:-1]) 

	sample_threshold = sample_freq * velocity_threshold / 1000
	
	x_coord = raw_fixations['x_cord']
	y_coord=raw_fixations['y_cord']
	gaze = np.stack([x_coord, y_coord])
	velocities = np.linalg.norm(gaze[:, 1:] - gaze[:, :-1], axis=0)
	velocities = np.concatenate([[0.], velocities])
	
	# Mark points below velocity threshold as fixations and the points above the threshold as saccades
	is_fixation = np.empty(len(x_coord), dtype=object)
	is_fixation[:] = True
	is_fixation[velocities > sample_threshold] = False

	# Define Viterbi sampler of the HMM and re­estimate fixation, saccade assignment for every eye position point
	
	# Use Baum­welch algorithm to re­estimate initial, transition,	and observation probabilities for the defined sampler
	# Merge Function(array of pre classified fixation and saccades)
	# Return saccades and fixations