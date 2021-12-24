import numpy as np

class getFixations:
    def getFixations(finalweights, timestamp, xpos, ypos, missing, par):
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
                    i = i-1
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
                fixend[p-1] = fixend[p]
                endtime[p-1]= endtime[p]
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

        # store all the results in a dictionary
        fix = {}
        fix['endT'] = endtime
        fix['dur'] = fixdur
        fix['xpos'] = xmedian
        fix['ypos'] = ymedian

        fix_list = list(fix.items())
        fix_arr = np.array(fix_list)

        return fix_arr.T