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

class NSLR_HMM_classifier:
        def nslr_hmm_classifier(raw_fixations,threshold,return_discrete=False):
            coord = np.vstack([raw_fixations[1],raw_fixations[2]]).T
            timestamp = raw_fixations[0]
            
            # classify using NSLR-HMM
            sample_class, seg, seg_class = nslr_hmm.classify_gaze(time_array, gaze_array,
                                                                **nslr_kwargs)
            
            # define discrete version of segments/classes
            segments = [s.t[0] for s in seg.segments]
            classes = seg_class
            
            # convert them if continuous series wanted
            if return_discrete == False:
                segments, classes = discrete_to_continuous(time_array, segments, classes)
            
            # add the prediction to our dataframe
            classes = [CLASSES[i] for i in classes]
            
            if return_orig_output:
                # create dictionary from it
                segment_dict = {"sample_class": sample_class, "segmentation": seg, "seg_class":seg_class}
                return segments, classes, segment_dict
            else:
                return segments, classes 


class IVT_classifier:
    def classify_velocity(raw_fixations,minimum_duration=50,sample_duration=4,velocity_threshold=0):
        """I-VT velocity algorithm from Salvucci & Goldberg (2000).
        
    
        For reference see:
        
        ---
        Salvucci, D. D., & Goldberg, J. H. (2000). Identifying fixations 
        and saccades in eye-tracking protocols. In Proceedings of the 
        2000 symposium on Eye tracking research & applications
        ---
        
        Parameters
		----------
		raw_fixations : list
			a list of fixations information containing timestamp, x_cord, and y_cord

		minimum_duration : int, optional
			minimum duration for a fixation in milliseconds, less than minimum is considered noise.
			set to 50 milliseconds by default

		sample_duration : int, optional
			Sample duration in milliseconds, this is 4 milliseconds based on this eye tracker

		velocity_threshold : int, optional
			maximum distance from a group of samples to be considered a single fixation.
			Set to 25 pixels by default

		Returns
		-------
		list
			a list where each element is a list of timestamp, duration, x_cord, and y_cord
		"""
        # process time argument and calculate sample threshold
       
        times=raw_fixations[0]
        sfreq = 1 / np.mean(times[1:] - times[:-1]) 

        sample_thresh = sfreq * velocity_threshold / 1000
        
        # calculate movement velocities
        x_coord=raw_fixations[1]
        y_coord=raw_fixations[2]
        coord = np.stack([x_coord, y_coord])
        vels = np.linalg.norm(gaze[:, 1:] - gaze[:, :-1], axis=0)
        vels = np.concatenate([[0.], vels])
        
        # define classes by threshold
        is_fixation = np.empty(len(x), dtype=object)
        is_fixation[:] = True
        is_fixation[vels > sample_thresh] = False
        
        # group consecutive classes to one segment
        segments = np.zeros(len(x), dtype=int)
        for idx in range(1, len(is_fixation)):
            if is_fixation[idx] == is_fixation[idx - 1]:
                segments[idx] = segments[idx - 1]
            else:
                segments[idx] = segments[idx - 1] + 1
        
        filter_fixation=[[time[segments[i]:segments[i+1]],(segments[i+1]-segments[i])*4,x_coord[segments[i]:segments[i+1]-1],y_coord[segments[i]:segments[i+1]-1]] for i in range(len(segments)-1)]
        return filter_fixation


class I2MC:
	def getFixations(finalweights, timestamp, xpos, ypos, missing, params):
		"""
		Description
		
		Parameters
		----------
		finalweights : type
			2-means clustering weighting
		timestamp : np.array
			Timestamp from Eyetracker (should be in ms!)
		xpos : np.array
			Horizontal coordinates from Eyetracker
		ypos : np.array
			Vertical coordinates from Eyetracker
		missing : np.array
			Vector containing the booleans for mising values
		par : Dictionary containing the following keys and values
			cutoffstd : float
				Number of std above mean clustering-weight to use as fixation cutoff
			onoffsetThresh : float
				Threshold (x*MAD of fixation) for walking forward/back for saccade off- and onsets
			maxMergeDist : float
				Maximum Euclidean distance in pixels between fixations for merging
			maxMergeTime : float
				Maximum time in ms between fixations for merging
			minFixDur : Float
				Minimum duration allowed for fiation
		Returns
		-------
		fix : Dictionary containing the following keys and values
			cutoff : float
				Cutoff used for fixation detection
			start : np.array
				Vector with fixation start indices
			end : np.array
				Vector with fixation end indices
			startT : np.array
				Vector with fixation start times
			endT : np.array
				Vector with fixation end times
			dur : type
				Vector with fixation durations
			xpos : np.array
				Vector with fixation median horizontal position (one value for each fixation in trial)
			ypos : np.array
				Vector with fixation median vertical position (one value for each fixation in trial)

		
		Examples
		--------
		>>> fix = getFixations(finalweights,data['time'],xpos,ypos,missing,par)
		>>> fix
			{'cutoff': 0.1355980099309374,
			'dur': array([366.599, 773.2  , 239.964, 236.608, 299.877, 126.637]),
			'end': array([111, 349, 433, 508, 600, 643]),
			'endT': array([ 369.919, 1163.169, 1443.106, 1693.062, 1999.738, 2142.977]),
			'flankdataloss': array([1., 0., 0., 0., 0., 0.]),
			'fracinterped': array([0.06363636, 0.        , 0.        , 0.        , 0.        ,
					0.        ]),
			'start': array([  2, 118, 362, 438, 511, 606]),
			'startT': array([   6.685,  393.325, 1206.498, 1459.79 , 1703.116, 2019.669]),
			'xpos': array([ 945.936,  781.056, 1349.184, 1243.92 , 1290.048, 1522.176]),
			'ypos': array([486.216, 404.838, 416.664, 373.005, 383.562, 311.904])}
		"""    
		### Extract the required parameters 
		cutoffstd = params['cutoffstd']
		onoffsetThresh = params['onoffsetThresh']
		maxMergeDist = params['maxMergeDist']
		maxMergeTime = params['maxMergeTime']
		minFixDur = params['minFixDur']
			
		### first determine cutoff for finalweights
		cutoff = np.nanmean(finalweights) + cutoffstd*np.nanstd(finalweights,ddof=1)

		### get boolean of fixations
		fixbool = finalweights < cutoff
		
		### get indices of where fixations start and end
		fixstart, fixend = bool2bounds(fixbool)
		
		### for each fixation start, walk forward until recorded position is below 
		# a threshold of lambda*MAD away from median fixation position.
		# same for each fixation end, but walk backward
		for p in range(len(fixstart)):
			xFix = xpos[fixstart[p]:fixend[p]+1]
			yFix = ypos[fixstart[p]:fixend[p]+1]
			xmedThis = np.nanmedian(xFix)
			ymedThis = np.nanmedian(yFix)
			
			# MAD = median(abs(x_i-median({x}))). For the 2D version, I'm using
			# median 2D distance of a point from the median fixation position. Not
			# exactly MAD, but makes more sense to me for 2D than city block,
			# especially given that we use 2D distance in our walk here
			MAD = np.nanmedian(np.hypot(xFix-xmedThis, yFix-ymedThis))
			thresh = MAD*onoffsetThresh

			# walk until distance less than threshold away from median fixation
			# position. No walking occurs when we're already below threshold.
			i = fixstart[p]
			if i>0:  # don't walk when fixation starting at start of data 
				while np.hypot(xpos[i]-xmedThis,ypos[i]-ymedThis)>thresh:
					i = i+1
				fixstart[p] = i
				
			# and now fixation end.
			i = fixend[p]
			if i<len(xpos): # don't walk when fixation ending at end of data
				while np.hypot(xpos[i]-xmedThis,ypos[i]-ymedThis)>thresh:
					i = i-1;
				fixend[p] = i

		### get start time, end time,
		starttime = timestamp[fixstart]
		endtime = timestamp[fixend]
		
		### loop over all fixation candidates in trial, see if should be merged
		for p in range(1,len(starttime))[::-1]:
			# get median coordinates of fixation
			xmedThis = np.median(xpos[fixstart[p]:fixend[p]+1])
			ymedThis = np.median(ypos[fixstart[p]:fixend[p]+1])
			xmedPrev = np.median(xpos[fixstart[p-1]:fixend[p-1]+1]);
			ymedPrev = np.median(ypos[fixstart[p-1]:fixend[p-1]+1]);
			
			# check if fixations close enough in time and space and thus qualify
			# for merging
			# The interval between the two fixations is calculated correctly (see
			# notes about fixation duration below), i checked this carefully. (Both
			# start and end of the interval are shifted by one sample in time, but
			# assuming practicalyl constant sample interval, thats not an issue.)
			if starttime[p]-endtime[p-1] < maxMergeTime and \
				np.hypot(xmedThis-xmedPrev,ymedThis-ymedPrev) < maxMergeDist:
				# merge
				fixend[p-1] = fixend[p];
				endtime[p-1]= endtime[p];
				# delete merged fixation
				fixstart = np.delete(fixstart, p)
				fixend = np.delete(fixend, p)
				starttime = np.delete(starttime, p)
				endtime = np.delete(endtime, p)
				
		### beginning and end of fixation must be real data, not interpolated.
		# If interpolated, those bit(s) at the edge(s) are excluded from the
		# fixation. First throw out fixations that are all missing/interpolated
		for p in range(len(starttime))[::-1]:
			miss = missing[fixstart[p]:fixend[p]+1]
			if np.sum(miss) == len(miss):
				fixstart = np.delete(fixstart, p)
				fixend = np.delete(fixend, p)
				starttime = np.delete(starttime, p)
				endtime = np.delete(endtime, p)
		
		# then check edges and shrink if needed
		for p in range(len(starttime)):
			if missing[fixstart[p]]:
				fixstart[p] = fixstart[p] + np.argmax(np.invert(missing[fixstart[p]:fixend[p]+1]))
				starttime[p]= timestamp[fixstart[p]]
			if missing[fixend[p]]:
				fixend[p] = fixend[p] - (np.argmax(np.invert(missing[fixstart[p]:fixend[p]+1][::-1]))+1)
				endtime[p] = timestamp[fixend[p]]
		
		### calculate fixation duration
		# if you calculate fixation duration by means of time of last sample during
		# fixation minus time of first sample during fixation (our fixation markers
		# are inclusive), then you always underestimate fixation duration by one
		# sample because you're in practice counting to the beginning of the
		# sample, not the end of it. To solve this, as end time we need to take the
		# timestamp of the sample that is one past the last sample of the fixation.
		# so, first calculate fixation duration by simple timestamp subtraction.
		fixdur = endtime-starttime
		
		# then determine what duration of this last sample was
		nextSamp = np.min(np.vstack([fixend+1,np.zeros(len(fixend),dtype=int)+len(timestamp)-1]),axis=0) # make sure we don't run off the end of the data
		extratime = timestamp[nextSamp]-timestamp[fixend] 
		
		# if last fixation ends at end of data, we need to determine how long that
		# sample is and add that to the end time. Here we simply guess it as the
		# duration of previous sample
		if not len(fixend)==0 and fixend[-1]==len(timestamp): # first check if there are fixations in the first place, or we'll index into non-existing data
			extratime[-1] = np.diff(timestamp[-3:-1])
		
		# now add the duration of the end sample to fixation durations, so we have
		# correct fixation durations
		fixdur = fixdur+extratime

		### check if any fixations are too short
		qTooShort = np.argwhere(fixdur<minFixDur)
		if len(qTooShort) > 0:
			fixstart = np.delete(fixstart, qTooShort)
			fixend = np.delete(fixend, qTooShort)
			starttime = np.delete(starttime, qTooShort)
			endtime = np.delete(endtime, qTooShort)
			fixdur = np.delete(fixdur, qTooShort)
			
		### process fixations, get other info about them
		xmedian = np.zeros(fixstart.shape) # vector for median
		ymedian = np.zeros(fixstart.shape)  # vector for median
		flankdataloss = np.zeros(fixstart.shape) # vector for whether fixation is flanked by data loss
		fracinterped = np.zeros(fixstart.shape) # vector for fraction interpolated
		for a in range(len(fixstart)):
			idxs = range(fixstart[a],fixend[a]+1)
			# get data during fixation
			xposf = xpos[idxs]
			yposf = ypos[idxs]
			# for all calculations below we'll only use data that is not
			# interpolated, so only real data
			qMiss = missing[idxs]
			
			# get median coordinates of fixation
			xmedian[a] = np.median(xposf[np.invert(qMiss)])
			ymedian[a] = np.median(yposf[np.invert(qMiss)])
			
			# determine whether fixation is flanked by period of data loss
			flankdataloss[a] = (fixstart[a]>0 and missing[fixstart[a]-1]) or (fixend[a]<len(xpos)-1 and missing[fixend[a]+1])
			
			# fraction of data loss during fixation that has been (does not count
			# data that is still lost)
			fracinterped[a]  = np.sum(np.invert(np.isnan(xposf[qMiss])))/(fixend[a]-fixstart[a]+1)

		# store all the results in an appropriate form 
		xpos_out=[xpos[fixstart[i]:fixend[i]] for i in range(len(fixstart))]
		ypos_out=[ypos[fixstart[i]:fixend[i]] for i in range(len(fixstart))]
		time_out=[ypos[fixstart[i]:fixend[i]] for i in range(len(fixstart))]

		return [time_out,xpos_out,ypos_out]


def I2MC(gazeData, options = {}):
    '''
    RUNS I2MC 
    
    
    Parameters
    ----------
    data : dict
        Dictionary containing all the data
    opt : dict
        Dictionary containing all the options 
    
    Returns
    -------
    fix : dict
        Dictionary containing all the fixation information
    
    Example
    --------
    >>> 
    '''
    # set defaults
    data = copy.deepcopy(gazeData)
    opt = options.copy()
    par = {}
    
    # Check required parameters 
    checkFun('xres', opt, 'horizontal screen resolution')
    checkFun('yres', opt, 'vertical screen resolution')
    checkFun('freq', opt, 'tracker sampling rate')
    checkFun('missingx', opt, 'value indicating data loss for horizontal position')
    checkFun('missingy', opt, 'value indicating data loss for vertical position')
    
    # required parameters:
    par['xres'] = opt.pop('xres')
    par['yres'] = opt.pop('yres')
    par['freq'] = opt.pop('freq')
    par['missingx'] = opt.pop('missingx')
    par['missingy'] = opt.pop('missingy')
    par['scrSz'] = opt.pop('scrSz', None ) # screen size (e.g. in cm). Optional, specify if want fixation statistics in deg
    par['disttoscreen'] = opt.pop('disttoscreen', None) # screen distance (in same unit as size). Optional, specify if want fixation statistics in deg
    
    #parameters with defaults:
    # CUBIC SPLINE INTERPOLATION
    par['windowtimeInterp'] = opt.pop('windowtimeInterp', .1) # max duration (s) of missing values for interpolation to occur
    par['edgeSampInterp'] = opt.pop('edgeSampInterp', 2) # amount of data (number of samples) at edges needed for interpolation
    par['maxdisp'] = opt.pop('maxdisp', None) # maximum displacement during missing for interpolation to be possible. Default set below if needed
    
    # K-MEANS CLUSTERING
    par['windowtime'] = opt.pop('windowtime', .2) # time window (s) over which to calculate 2-means clustering (choose value so that max. 1 saccade can occur)
    par['steptime'] = opt.pop('steptime', .02) # time window shift (s) for each iteration. Use zero for sample by sample processing
    par['downsamples'] = opt.pop('downsamples', [2, 5, 10]) # downsample levels (can be empty)
    par['downsampFilter'] = opt.pop('downsampFilter', True) # use chebychev filter when downsampling? True: yes, False: no. requires signal processing toolbox. is what matlab's downsampling functions do, but could cause trouble (ringing) with the hard edges in eye-movement data
    par['chebyOrder'] = opt.pop('chebyOrder', 8.) # order of cheby1 Chebyshev downsampling filter, default is normally ok, as long as there are 25 or more samples in the window (you may have less if your data is of low sampling rate or your window is small
    par['maxerrors'] = opt.pop('maxerrors', 100.) # maximum number of errors allowed in k-means clustering procedure before proceeding to next file
    # FIXATION DETERMINATION
    par['cutoffstd'] = opt.pop('cutoffstd', 2.) # number of standard deviations above mean k-means weights will be used as fixation cutoff
    par['onoffsetThresh']  = opt.pop('onoffsetThresh', 3.) # number of MAD away from median fixation duration. Will be used to walk forward at fixation starts and backward at fixation ends to refine their placement and stop algorithm from eating into saccades
    par['maxMergeDist'] = opt.pop('maxMergeDist', 30.) # maximum Euclidean distance in pixels between fixations for merging
    par['maxMergeTime'] = opt.pop('maxMergeTime', 30.) # maximum time in ms between fixations for merging
    par['minFixDur'] = opt.pop('minFixDur', 40.) # minimum fixation duration (ms) after merging, fixations with shorter duration are removed from output
      
    # Development parameters (plotting intermediate steps), Change these to False when not developing
    par['dev_interpolation'] = opt.pop('dev_interpolation', False)
    par['dev_cluster'] = opt.pop('dev_cluster', False)
    par['skip_inputhandeling'] = opt.pop('skip_inputhandeling', False)

    for key in opt:
        assert False, 'Key "{}" not recognized'.format(key)
    
    # =============================================================================
    # # Input handeling and checking
    # =============================================================================
    ## loop over input
    if not par['skip_inputhandeling']:
        for key, value in par.items():
            if key in ['xres','yres','freq','missingx','missingy','windowtimeInterp','maxdisp','windowtime','steptime','cutoffstd','onoffsetThresh','maxMergeDist','maxMergeTime','minFixDur']:
                checkNumeric(key,value)
                checkScalar(key,value)
            elif key == 'disttoscreen':
                if value is not None:   # may be none (its an optional parameter)
                    checkNumeric(key,value)
                    checkScalar(key,value)
            elif key in ['downsampFilter','chebyOrder','maxerrors','edgeSampInterp']:
                checkInt(key,value)
                checkScalar(key,value)
            elif key == 'scrSz':
                if value is not None:   # may be none (its an optional parameter)
                    checkNumeric(key,value)
                    checkNumel2(key,value)
            elif key == 'downsamples':
                checkInt(key,value)
            else:
                if type(key) != str:
                    assert False, 'Key "{}" not recognized'.format(key)
    
    # set defaults
    if par['maxdisp'] is None:
        par['maxdisp'] = par['xres']*0.2*np.sqrt(2)

    # check filter
    if par['downsampFilter']:
        nSampRequired = np.max([1,3*par['chebyOrder']])+1  # nSampRequired = max(1,3*(nfilt-1))+1, where nfilt = chebyOrder+1
        nSampInWin = round(par['windowtime']/(1./par['freq']))
        assert nSampInWin>=nSampRequired,'I2MCfunc: Filter parameters requested with the setting ''chebyOrder'' will not work for the sampling frequency of your data. Please lower ''chebyOrder'', or set the setting ''downsampFilter'' to 0'
   
    assert np.sum(par['freq']%np.array(par['downsamples'])) ==0,'I2MCfunc: Some of your downsample levels are not divisors of your sampling frequency. Change the option "downsamples"'
    
   

   
    data['finalweights'], stopped = twoClusterWeighting(xpos, ypos, missingn, par['downsamples'], par['downsampFilter'], par['chebyOrder'],par['windowtime'], par['steptime'],par['freq'],par['maxerrors'],par['dev_cluster'])
        
        # check whether clustering succeeded
    if stopped:
        print('\tClustering stopped after exceeding max errors, continuing to next file \n')
        return False
        
    # =============================================================================
    #  DETERMINE FIXATIONS BASED ON FINALWEIGHTS_AVG
    # =============================================================================
    print('\tDetermining fixations based on clustering weight mean for averaged signal and separate eyes + {:.2f}*std'.format(par['cutoffstd']))
    fix = getFixations(data['finalweights'],data['time'],xpos,ypos,missing,par)
  
    return fix