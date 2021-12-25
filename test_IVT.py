import numpy as np
from Classifier import IVT
def test_velocity():
	ivt=IVT()
	x_coord=np.array([1,5,9])
	y_coord=np.array([1,4,7])
	times=np.array([0,2,4])
	assert np.all(ivt.velocity(x_coord,y_coord,times)==np.array([2.5,2.5]))

def test_classify():
	ivt=IVT()

	raw_fixations=[[1,2,3],[4,5,6],[7,8,9]]
	raw_fixations_np=np.array(raw_fixations)
	threshold=1.5
	times=raw_fixations_np[:,0]


	#first test is to make sure that is_fixation works: 
	# calculate movement velocities
	x_cord=raw_fixations_np[:,1]
	y_cord=raw_fixations_np[:,2]
	vels=ivt.velocity(x_cord,y_cord,times)
	# define classes by threshold
	is_fixation = np.empty(len(x_cord)-1, dtype=object)
	is_fixation[:] = True
	is_fixation[vels > threshold ] = False
	assert np.all(is_fixation)

	#2nd test is for the last component of classify(), which identify whether it is fixation. We use sample data for convenience
	vels=np.array([1,2,3,4,10,11,12,1,2,3,4,10,11,12])
	x_cord=np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
	y_cord=np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
	times=np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])

	is_fixation = np.empty(len(vels), dtype=object)
	is_fixation[:] = True
	is_fixation[vels > 5] = False
	assert np.all(is_fixation==np.array([True,True,True,True,False,False,False,True,True,True,True,False,False,False]))
	segments = np.zeros(len(vels), dtype=int)
	
	for i in range(1, len(is_fixation)):
		if is_fixation[i] == is_fixation[i - 1]:
			segments[i] = segments[i - 1]
		else:
			segments[i] = segments[i - 1] + 1
	assert np.all(segments==np.array([0,0,0,0,1,1,1,2,2,2,2,3,3,3]))

	#3rd test is to check that the format of segments are all correct for fixation!
	filter_fixation=[]
	# go through each segment, if the segment only lasts for 50ms, ignore, otherwise add it to the list 
	assert np.max(segments)==3
	for i in range (0,np.max(segments)+1):
		segment_where=np.where(segments==i)[0]
		if segment_where.shape[0]<2 or np.all(is_fixation[segment_where[0]]==False):
			#either too small of a segment, or it is not a fixation segment:
			assert i==1 or i==3
			continue
		assert i==0 or i==2
		time_now=times[segment_where[-1]]
		
		duration_now=len(segment_where)*4
		x_cord_now=np.mean(x_cord[segment_where])
		y_cord_now=np.mean(y_cord[segment_where])
		filter_fixation.append([time_now,duration_now,x_cord_now,y_cord_now])

	#4th test for output:
	filter_fixation=np.array(filter_fixation)	
	assert np.all(filter_fixation==np.array([[4,16.,2.5,2.5],[11.,16.,9.5,9.5]]))
def main():
	test_velocity()
	test_classify()
if __name__=="__main__":
	main()