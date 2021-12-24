'''
Sidney Liu
Helper Functions Refactored for I2MC Class
CS321
'''

import numpy as np
import scipy

class twoClusterWeighting:
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


            if off[-1]>(len(xpos)-1-nrsamples):
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
        while eind<=(len(xpos)-1):
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
                
            if eind>len(xpos)-1 and eind-stepsize<len(xpos)-1:
                # we just exceeded data bound, but previous eind was before end of
                # data: we have some unprocessed samples. retreat just enough so we
                # process those end samples once
                d = eind-len(xpos)+1
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
                lonely = np.argwhere(label==cFrom)[0]
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