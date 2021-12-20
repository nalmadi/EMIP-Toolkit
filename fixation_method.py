import numpy as np
import math
import scipy
import scipy.interpolate as interp
import scipy.signal
from scipy.cluster.vq import vq, _vq
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as patches
import copy
import warnings

class EMIP_filter:
    def idt_classifier(raw_fixations, minimum_duration=50, sample_duration=4, maximum_dispersion=25):
            """I-DT classifier based on page 296 of eye tracker manual:
                https://psychologie.unibas.ch/fileadmin/user_upload/psychologie/Forschung/N-Lab/SMI_iView_X_Manual.pdf

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

            # Create moving window based on minimum_duration
            window_size = int(math.ceil(minimum_duration / sample_duration))

            window_x = []
            window_y = []

            filter_fixation = []

            # Go over all SMPs in trial data
            for timestamp, x_cord, y_cord in raw_fixations:

                # Filter (skip) coordinates outside of the screen 1920×1080 px
                if x_cord < 0 or y_cord < 0 or x_cord > 1920 or y_cord > 1080:
                    continue

                # Add sample if it appears to be valid
                window_x.append(x_cord)
                window_y.append(y_cord)

                # Calculate dispersion = [max(x) - min(x)] + [max(y) - min(y)]
                dispersion = (max(window_x) - min(window_x)) + (max(window_y) - min(window_y))

                # If dispersion is above maximum_dispersion
                if dispersion > maximum_dispersion:

                    # Then the window does not represent a fixation
                    # Pop last item in window
                    window_x.pop()
                    window_y.pop()

                    # Add fixation to fixations if window is not empty (size >= window_size)
                    if len(window_x) == len(window_y) and len(window_x) > window_size:
                        # The fixation is registered at the centroid of the window points
                        filter_fixation.append(
                            [timestamp, len(window_x) * 4, statistics.mean(window_x), statistics.mean(window_y)])

                    window_x = []
                    window_y = []

            return filter_fixation
    



    def bool2bounds(b):
        """
        Finds all contiguous sections of true in a boolean
        Parameters
        ----------
        data : np.array
            A 1d np.array containing True, False values.
        
        Returns
        -------
        on : np.array
            The array contains the indexes of the first value = True
        off : np.array
            The array contains the indexes of the last value = True in a sequence
        
        Example
        --------
        >>> import numpy as np
        >>> b = np.array([1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0])
        >>> on, off = bool2bounds(b)
        >>> print(on)
        [0 4 8]
        >>> print(off)
        [0 6 9]
        """
        b = np.array(np.array(b, dtype = np.bool), dtype=int)
        b = np.pad(b, (1, 1), 'constant', constant_values=(0, 0))
        D = np.diff(b)
        on  = np.array(np.where(D == 1)[0], dtype=int)
        off = np.array(np.where(D == -1)[0] -1, dtype=int)
        return on, off
    def twoClusterWeighting(xpos, ypos, missing, downsamples, downsampFilter, chebyOrder, windowtime, steptime, freq, maxerrors, dev=False):
        """
        Description
        
        Parameters
        ----------
        xpos : type
            Description
        ypos : type
            Description
        missing : type
            Description
        downsamples : type
            Description
        downsampFilter : type
            Description
        chebyOrder : type
            Description
        windowtime : type
            Description
        steptime : type
            Description
        freq : type
            Description
        maxerrors : type
            Description
        Returns
        -------
        finalweights : np.array
            Vector of 2-means clustering weights (one weight for each sample), the higher, the more likely a saccade happened        
            
        Examples
        --------
        >>> 
        """   
        # calculate number of samples of the moving window
        nrsamples = int(windowtime/(1./freq))
        stepsize  = np.max([1,int(steptime/(1./freq))])
        
        # create empty weights vector
        totalweights = np.zeros(len(xpos))
        totalweights[missing] = np.nan
        nrtests = np.zeros(len(xpos))
        
        # stopped is always zero, unless maxiterations is exceeded. this
        # indicates that file could not be analysed after trying for x iterations
        stopped = False
        counterrors = 0
        
        # Number of downsamples
        nd = len(downsamples)
        
        # Downsample 
        if downsampFilter:
            # filter signal. Follow the lead of decimate(), which first runs a
            # Chebychev filter as specified below
            rp = .05 # passband ripple in dB
            b = [[] for i in range(nd)]
            a = [[] for i in range(nd)]
            for p in range(nd):
                b[p],a[p] = scipy.signal.cheby1(chebyOrder, rp, .8/downsamples[p]) 
        
        
        # idx for downsamples
        idxs = []
        for i in range(nd):
            idxs.append(np.arange(nrsamples,0,-downsamples[i],dtype=int)[::-1] - 1)
            
        # see where are missing in this data, for better running over the data
        # below.
        on,off = bool2bounds(missing)
        if on.size > 0:
            #  merge intervals smaller than nrsamples long 
            merge = np.argwhere((on[1:] - off[:-1])-1 < nrsamples).flatten()
            for p in merge[::-1]:
                off[p] = off[p+1]
                off = np.delete(off, p+1)
                on = np.delete(on, p+1)

            # check if intervals at data start and end are large enough
            if on[0]<nrsamples+1:
                # not enough data point before first missing, so exclude them all
                on[0]=0

            if off[-1]>(len(xpos)-nrsamples):
                # not enough data points after last missing, so exclude them all
                off[-1]=len(xpos)-1

            # start at first non-missing sample if trial starts with missing (or
            # excluded because too short) data
            if on[0]==0:
                i=off[0]+1 # start at first non-missing
            else:
                i=0
        else:
            i=0

        eind = i+nrsamples
        while eind<=(len(xpos)):
            # check if max errors is crossed
            if counterrors > maxerrors:
                print('Too many empty clusters encountered, aborting file. \n')
                stopped = True
                finalweights = np.nan
                return finalweights, stopped
            
            # select data portion of nrsamples
            idx = range(i,eind)
            ll_d = [[] for p in range(nd+1)]
            IDL_d = [[] for p in range(nd+1)]
            ll_d[0] = np.vstack([xpos[idx], ypos[idx]])
                    
            # Filter the bit of data we're about to downsample. Then we simply need
            # to select each nth sample where n is the integer factor by which
            # number of samples is reduced. select samples such that they are till
            # end of window
            for p in range(nd):
                if downsampFilter:
                    ll_d[p+1] = scipy.signal.filtfilt(b[p],a[p],ll_d[0])
                    ll_d[p+1] = ll_d[p+1][:,idxs[p]]
                else:
                    ll_d[p+1] = ll_d[0][:,idxs[p]]
            
            # do 2-means clustering
            try:
                for p in range(nd+1):
                    IDL_d[p] = kmeans2(ll_d[p].T)[0]
            except Exception as e:
                print('Unknown error encountered at sample {}.\n'.format(i))
                raise e
            
            # detect switches and weight of switch (= 1/number of switches in
            # portion)
            switches = [[] for p in range(nd+1)]
            switchesw = [[] for p in range(nd+1)]
            for p in range(nd+1):
                switches[p] = np.abs(np.diff(IDL_d[p]))
                switchesw[p]  = 1./np.sum(switches[p])
            
            # get nearest samples of switch and add weight
            weighted = np.hstack([switches[0]*switchesw[0],0])
            for p in range(nd):
                j = np.array((np.argwhere(switches[p+1]).flatten()+1)*downsamples[p],dtype=int)-1
                for o in range(int(downsamples[p])):
                    weighted[j+o] = weighted[j+o] + switchesw[p+1]
            
            # add to totalweights
            totalweights[idx] = totalweights[idx] + weighted
            # record how many times each sample was tested
            nrtests[idx] = nrtests[idx] + 1
            
            # update i
            i += stepsize
            eind += stepsize
            missingOn = np.logical_and(on>=i, on<=eind)
            missingOff = np.logical_and(off>=i, off<=eind)
            qWhichMiss = np.logical_or(missingOn, missingOff) 
            if np.sum(qWhichMiss) > 0:
                # we have some missing in this window. we don't process windows
                # with missing. Move back if we just skipped some samples, or else
                # skip whole missing and place start of window and first next
                # non-missing.
                if on[qWhichMiss][0] == (eind-stepsize):
                    # continue at first non-missing
                    i = off[qWhichMiss][0]+1
                else:
                    # we skipped some points, move window back so that we analyze
                    # up to first next missing point
                    i = on[qWhichMiss][0]-nrsamples
                eind = i+nrsamples
                
            if eind>len(xpos) and eind-stepsize<len(xpos):
                # we just exceeded data bound, but previous eind was before end of
                # data: we have some unprocessed samples. retreat just enough so we
                # process those end samples once
                d = eind-len(xpos)
                eind = eind-d
                i = i-d
                

        # create final weights
        finalweights = totalweights/nrtests
        
        return finalweights, stopped


    def kmeans2(data):
        # n points in p dimensional space
        n = data.shape[0]

        maxit = 100

        ## initialize using kmeans++ method.
        # code taken and slightly edited from scipy.cluster.vq
        dims = data.shape[1] if len(data.shape) > 1 else 1
        C = np.ndarray((2, dims))
        
        # first cluster
        C[0, :] = data[np.random.randint(data.shape[0])]

        # second cluster
        D = cdist(C[:1,:], data, metric='sqeuclidean').min(axis=0)
        probs = D/D.sum()
        cumprobs = probs.cumsum()
        r = np.random.rand()
        C[1, :] = data[np.searchsorted(cumprobs, r)]

        # Compute the distance from every point to each cluster centroid and the
        # initial assignment of points to clusters
        D = cdist(C, data, metric='sqeuclidean')
        # Compute the nearest neighbor for each obs using the current code book
        label = vq(data, C)[0]
        # Update the code book by computing centroids
        C = _vq.update_cluster_means(data, label, 2)[0]
        m = np.bincount(label)

        ## Begin phase one:  batch reassignments
        #-----------------------------------------------------
        # Every point moved, every cluster will need an update
        prevtotsumD = math.inf
        iter = 0
        while True:
            iter += 1
            # Calculate the new cluster centroids and counts, and update the
            # distance from every point to those new cluster centroids
            Clast = C
            mlast = m
            D = cdist(C, data, metric='sqeuclidean')

            # Deal with clusters that have just lost all their members
            if np.any(m==0):
                i = np.argwhere(m==0)
                d = D[[label],[range(n)]]   # use newly updated distances
            
                # Find the point furthest away from its current cluster.
                # Take that point out of its cluster and use it to create
                # a new singleton cluster to replace the empty one.
                lonely = np.argmax(d)
                cFrom = label[lonely]    # taking from this cluster
                if m[cFrom] < 2:
                    # In the very unusual event that the cluster had only
                    # one member, pick any other non-singleton point.
                    cFrom = np.argwhere(m>1)[0]
                    lonely = np.argwhere(mlabel==cFrom)[0]
                end
                label[lonely] = i
            
                # Update clusters from which points are taken
                C = _vq.update_cluster_means(data, label, 2)[0]
                m = np.bincount(label)
                D = cdist(C, data, metric='sqeuclidean')
        
            # Compute the total sum of distances for the current configuration.
            totsumD = np.sum(D[[label],[range(n)]])
            # Test for a cycle: if objective is not decreased, back out
            # the last step and move on to the single update phase
            if prevtotsumD <= totsumD:
                label = prevlabel
                C = Clast
                m = mlast
                iter -= 1
                break
            if iter >= maxit:
                break
        
            # Determine closest cluster for each point and reassign points to clusters
            prevlabel = label
            prevtotsumD = totsumD
            newlabel = vq(data, C)[0]
        
            # Determine which points moved
            moved = newlabel != prevlabel
            if np.any(moved):
                # Resolve ties in favor of not moving
                moved[np.bitwise_and(moved, D[0,:]==D[1,:])] = False
            if not np.any(moved):
                break
            label = newlabel
            # update centers
            C = _vq.update_cluster_means(data, label, 2)[0]
            m = np.bincount(label)


        #------------------------------------------------------------------
        # Begin phase two:  single reassignments
        #------------------------------------------------------------------
        lastmoved = -1
        converged = False
        while iter < maxit:
            # Calculate distances to each cluster from each point, and the
            # potential change in total sum of errors for adding or removing
            # each point from each cluster.  Clusters that have not changed
            # membership need not be updated.
            #
            # Singleton clusters are a special case for the sum of dists
            # calculation. Removing their only point is never best, so the
            # reassignment criterion had better guarantee that a singleton
            # point will stay in its own cluster. Happily, we get
            # Del(i,idx(i)) == 0 automatically for them.
            Del = cdist(C, data, metric='sqeuclidean')
            mbrs = label==0
            sgn = 1 - 2*mbrs    # -1 for members, 1 for nonmembers
            if m[0] == 1:
                sgn[mbrs] = 0   # prevent divide-by-zero for singleton mbrs
            Del[0,:] = (m[0] / (m[0] + sgn)) * Del[0,:]
            # same for cluster 2
            sgn = -sgn          # -1 for members, 1 for nonmembers
            if m[1] == 1:
                sgn[np.invert(mbrs)] = 0    # prevent divide-by-zero for singleton mbrs
            Del[1,:] = (m[1] / (m[1] + sgn)) * Del[1,:]
        
            # Determine best possible move, if any, for each point.  Next we
            # will pick one from those that actually did move.
            prevlabel = label
            newlabel = (Del[1,:]<Del[0,:]).astype('int')
            moved = np.argwhere(prevlabel != newlabel)
            if moved.size>0:
                # Resolve ties in favor of not moving
                moved = np.delete(moved,(Del[0,moved]==Del[1,moved]).flatten(),None)
            if moved.size==0:
                converged = True
                break
        
            # Pick the next move in cyclic order
            moved = (np.min((moved - lastmoved % n) + lastmoved) % n)
        
            # If we've gone once through all the points, that's an iteration
            if moved <= lastmoved:
                iter = iter + 1
                if iter >= maxit:
                    break
            lastmoved = moved
        
            olbl = label[moved]
            nlbl = newlabel[moved]
            totsumD = totsumD + Del[nlbl,moved] - Del[olbl,moved]
        
            # Update the cluster index vector, and the old and new cluster
            # counts and centroids
            label[moved] = nlbl
            m[nlbl] += 1
            m[olbl] -= 1
            C[nlbl,:] = C[nlbl,:] + (data[moved,:] - C[nlbl,:]) / m[nlbl]
            C[olbl,:] = C[olbl,:] - (data[moved,:] - C[olbl,:]) / m[olbl]
        
        #------------------------------------------------------------------
        if not converged:
            warnings.warn("kmeans failed to converge after {} iterations".format(maxit))

        return label, C
    def getFixations(finalweights, raw_fixations, missing, par):
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
            flankdataloss : bool
                Boolean with 1 for when fixation is flanked by data loss, 0 if not flanked by data loss
            fracinterped : float
                Fraction of data loss/interpolated data
        
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
        cutoffstd = par['cutoffstd']
        onoffsetThresh = par['onoffsetThresh']
        maxMergeDist = par['maxMergeDist']
        maxMergeTime = par['maxMergeTime']
        minFixDur = par['minFixDur']
            
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
            xmedPrev = np.median(xpos[fixstart[p-1]:fixend[p-1]+1])
            ymedPrev = np.median(ypos[fixstart[p-1]:fixend[p-1]+1])
            
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

        # store all the results in a dictionary
        filter_fixation =[]
        filter_fixation['cutoff'] = cutoff
        filter_fixation['start'] = fixstart
        filter_fixation['end'] = fixend
        filter_fixation['startT'] = starttime
        filter_fixation['endT'] = endtime
        filter_fixation['dur'] = fixdur
        filter_fixation['xpos'] = xmedian
        filter_fixation['ypos'] = ymedian
        filter_fixation['flankdataloss'] = flankdataloss
        filter_fixation['fracinterped'] = fracinterped
        return fix

    def getFixStats(xpos, ypos, missing, pixperdeg = None, fix = {}):
        """
        Description
        
        Parameters
        ----------
        xpos : np.array
            X gaze positions
        ypos : np.array
            Y gaze positions
        missing : np.array - Boolean
            Vector containing the booleans for mising values (originally, before interpolation!) 
        pixperdeg : float
            Number of pixels per visual degree
        fix : Dictionary containing the following keys and values
            fstart : np.array
                fixation start indices
            fend : np.array
                fixation end indices
        Returns
        -------
        fix : the fix input dictionary with the following added keys and values 
            RMSxy : float
                RMS of fixation (precision)
            BCEA : float 
                BCEA of fixation (precision)
            rangeX : float
                max(xpos) - min(xpos) of fixation
            rangeY : float
                max(ypos) - min(ypos) of fixation
            
        Examples
        --------
        >>> fix = getFixStats(xpos,ypos,missing,fix,pixperdeg)
        >>> fix
            {'BCEA': array([0.23148877, 0.23681681, 0.24498942, 0.1571361 , 0.20109245,
                0.23703843]),
        'RMSxy': array([0.2979522 , 0.23306149, 0.27712236, 0.26264146, 0.28913117,
                0.23147076]),
        'cutoff': 0.1355980099309374,
        'dur': array([366.599, 773.2  , 239.964, 236.608, 299.877, 126.637]),
        'end': array([111, 349, 433, 508, 600, 643]),
        'endT': array([ 369.919, 1163.169, 1443.106, 1693.062, 1999.738, 2142.977]),
        'fixRangeX': array([0.41066299, 0.99860672, 0.66199772, 0.49593727, 0.64628929,
                0.81010568]),
        'fixRangeY': array([1.58921528, 1.03885955, 1.10576059, 0.94040142, 1.21936613,
                0.91263117]),
        'flankdataloss': array([1., 0., 0., 0., 0., 0.]),
        'fracinterped': array([0.06363636, 0.        , 0.        , 0.        , 0.        ,
                0.        ]),
        'start': array([  2, 118, 362, 438, 511, 606]),
        'startT': array([   6.685,  393.325, 1206.498, 1459.79 , 1703.116, 2019.669]),
        'xpos': array([ 945.936,  781.056, 1349.184, 1243.92 , 1290.048, 1522.176]),
        'ypos': array([486.216, 404.838, 416.664, 373.005, 383.562, 311.904])}
        """
        ### Extract the required parameters 
        fstart = fix['start']
        fend = fix['end']

        # vectors for precision measures
        RMSxy = np.zeros(fstart.shape)
        BCEA  = np.zeros(fstart.shape)
        rangeX = np.zeros(fstart.shape)
        rangeY = np.zeros(fstart.shape)

        for a in range(len(fstart)):
            idxs = range(fstart[a],fend[a]+1)
            # get data during fixation
            xposf = xpos[idxs]
            yposf = ypos[idxs]
            # for all calculations below we'll only use data that is not
            # interpolated, so only real data
            qMiss = missing[idxs]
            
            ### calculate RMS
            # since its done with diff, don't just exclude missing and treat
            # resulting as one continuous vector. replace missing with nan first,
            # use left-over values
            # Difference x position
            xdif = xposf.copy()
            xdif[qMiss] = np.nan
            xdif = np.diff(xdif)**2; 
            xdif = xdif[np.invert(np.isnan(xdif))]
            # Difference y position
            ydif = yposf.copy()
            ydif[qMiss] = np.nan
            ydif = np.diff(ydif)**2; 
            ydif = ydif[np.invert(np.isnan(ydif))]
            # Distance and RMS measure
            c = xdif + ydif # 2D sample-to-sample displacement value in pixels
            RMSxy[a] = np.sqrt(np.mean(c))
            if pixperdeg is not None:
                RMSxy[a] = RMSxy[a]/pixperdeg # value in degrees visual angle
            
            ### calculate BCEA (Crossland and Rubin 2002 Optometry and Vision Science)
            stdx = np.std(xposf[np.invert(qMiss)],ddof=1)
            stdy = np.std(yposf[np.invert(qMiss)],ddof=1)
            if pixperdeg is not None:
                # value in degrees visual angle
                stdx = stdx/pixperdeg
                stdy = stdy/pixperdeg
        
            if len(yposf[np.invert(qMiss)])<2:
                BCEA[a] = np.nan
            else:
                xx = np.corrcoef(xposf[np.invert(qMiss)],yposf[np.invert(qMiss)])
                rho = xx[0,1]
                P = 0.68 # cumulative probability of area under the multivariate normal
                k = np.log(1./(1-P))
                BCEA[a] = 2*k*np.pi*stdx*stdy*np.sqrt(1-rho**2);
            
            ### calculate max-min of fixation
            if np.sum(qMiss) == len(qMiss):
                rangeX[a] = np.nan
                rangeY[a] = np.nan
            else:
                rangeX[a] = (np.max(xposf[np.invert(qMiss)]) - np.min(xposf[np.invert(qMiss)]))
                rangeY[a] = (np.max(yposf[np.invert(qMiss)]) - np.min(yposf[np.invert(qMiss)]))

            if pixperdeg is not None:
                # value in degrees visual angle
                rangeX[a] = rangeX[a]/pixperdeg;
                rangeY[a] = rangeY[a]/pixperdeg;

        # Add results to fixation dictionary
        fix['RMSxy'] = RMSxy
        fix['BCEA'] = BCEA
        fix['fixRangeX'] = rangeX
        fix['fixRangeY'] = rangeY
        
        return fix

    def I2MC_classifier(gazeData, options = {}):
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
        
        # =============================================================================
        # 2-MEANS CLUSTERING
        # =============================================================================
        ## CALCULATE 2-MEANS CLUSTERING FOR SINGLE EYE
        # get kmeans-clustering for averaged signal
        print('\t2-Means clustering started for averaged signal')
        data['finalweights'], stopped = twoClusterWeighting(raw_fixations, mising, par['downsamples'], par['downsampFilter'], par['chebyOrder'],par['windowtime'], par['steptime'],par['freq'],par['maxerrors'],par['dev_cluster'])
            
        # check whether clustering succeeded
        if stopped:
            print('\tClustering stopped after exceeding max errors, continuing to next file \n')
            return False
    
        # =============================================================================
        #  DETERMINE FIXATIONS BASED ON FINALWEIGHTS_AVG
        # =============================================================================
        print('\tDetermining fixations based on clustering weight mean for averaged signal and separate eyes + {:.2f}*std'.format(par['cutoffstd']))
        fix = getFixations(data['finalweights'],data['time'],xpos,ypos,missing,par)
        fix = getFixStats(xpos,ypos,missing,pixperdeg,fix)
        
        return fix,data,par