'''
This file will take in <new_video_mp4> absolute path and <practice_videos_folder> absolute path.
It will use Dynamic Time Warp to compare the normed wrist coords of the new video comared to 
all of the practice videos. It will keep track of the k closest practice videos to the new video.
Returns: The mode category from the top k closest practice videos as a category.txt file.
Ex:
python3 gesture_id.py /home/ubuntu/HandGesturesPracticeVideos/ /home/ubuntu/NewVideos/Email_PRACTICE_2_Manuszak.mp4
'''


# Running Dynamic Time Warping
# Reference: https://dynamictimewarping.github.io/python/

import csv
import numpy as np
import matplotlib.pyplot as plt
from dtw import *
import sys
from os import listdir
from os.path import isfile, isdir, join, splitext
from statistics import mode

practice_videos_folder = sys.argv[1]
new_video_mp4 = sys.argv[2]
k = 15  # This determines the top k closest practice gestures to this new video

##### GETTING THE NORMED ARRAY OF THE NEW VIDEO #####
# Initializations
left_wrist_X_raw = 0
left_wrist_Y_raw = 0
right_wrist_X_raw = 0
right_wrist_Y_raw = 0

# Normalizer values - shoulder, nose, hip
nose_X, nose_Y = 0, 0
left_shoulder_X, right_shoulder_X = 0, 0
right_hip_Y, left_hip_Y = 0, 0

right_wrist_X_norm, right_wrist_Y_norm = [], []
left_wrist_X_norm, left_wrist_Y_norm = [], []

with open(splitext(new_video_mp4)[0] + "/key_points.csv") as csv_file:
    raw_data = csv.reader(csv_file, delimiter=',')
    line = 0
    # Data extraction
    for row in raw_data:
      if not line:
        line = line+1
        continue
      line = line+1
      left_wrist_X_raw = float(row[30])
      left_wrist_Y_raw = float(row[31])
      right_wrist_X_raw = float(row[33])
      right_wrist_Y_raw = float(row[34])
      nose_X = float(row[3])
      nose_Y = float(row[4])
      left_shoulder_X = float(row[18])
      right_shoulder_X = float(row[21])
      right_hip_Y = float(row[40])
      left_hip_Y = float(row[37])

      right_wrist_X_norm.append((right_wrist_X_raw - nose_X) / abs(left_shoulder_X - right_shoulder_X))
      right_wrist_Y_norm.append((right_wrist_Y_raw - nose_Y) / abs(nose_Y - (right_hip_Y+left_hip_Y)/2))
      left_wrist_X_norm.append((left_wrist_X_raw - nose_X) / abs(left_shoulder_X - right_shoulder_X))
      left_wrist_Y_norm.append((left_wrist_X_raw - nose_Y) / abs(nose_Y - (right_hip_Y+left_hip_Y)/2))
      
# These are the normed arrays of the new video
right_wrist_X_norm = np.asarray(right_wrist_X_norm)
right_wrist_Y_norm = np.asarray(right_wrist_Y_norm)
left_wrist_X_norm = np.asarray(left_wrist_X_norm)
left_wrist_Y_norm = np.asarray(left_wrist_Y_norm)

#### GETTING THE TOP K CLOSEST PRACTICE VIDEO CATEGORIES ####
'''
For each video in the practice_videos_folder, 
 - load the normed arrays of the practice video
 - Measure the DTW of each array from the new video to the practice video pair-wise
 - Add up all of these measures..
   + This sum will be the measure for how close this new video is to the practice video.
 - If it falls in the top k closest, replace with the largest currently in top k
When all done, output the majority category belonging to the top k.
'''

# This is an array of 2d tuples, where tuple[0] is the practice video name
# and tuple[1] is the distance from the new video to tuple[0] practice video
# The length of this is at most k.
topk = []

for directory in [d for d in listdir(practice_videos_folder) if isdir(join(practice_videos_folder, d))]:
	# Arrays with prefix p are for the practice video
  try:
		p_right_wrist_X_norm = np.load(practice_videos_folder + directory + "/right_wrist_X_norm.npy")
		p_right_wrist_Y_norm = np.load(practice_videos_folder + directory + "/right_wrist_Y_norm.npy")
		p_left_wrist_X_norm = np.load(practice_videos_folder + directory + "/left_wrist_X_norm.npy")
		p_left_wrist_Y_norm = np.load(practice_videos_folder + directory + "/left_wrist_Y_norm.npy")
		p_video_norm_arrs = [p_right_wrist_X_norm, p_right_wrist_Y_norm, p_left_wrist_X_norm, p_left_wrist_Y_norm]

		tup = (directory, sum([dtw(new_video_norm_arrs[i], p_video_norm_arrs[i]).distance for i in range(len(p_video_norm_arrs))]))
		topk.append(tup)
		topk.sort(key = lambda x: x[1], reverse=True)
		if (len(topk) > k):
			topk = topk[1:]
	except Exception as e:
		print("An error occured while trying to DTW with " + directory)
		print(e)

categories = []
for tup in topk:
	with open(practice_videos_folder + tup[0] + "/category.txt") as f:
		categories.append(f.readline().rstrip())
with open(splitext(new_video_mp4)[0] + "/category_lamba.txt", "w") as f:
	f.write(mode(categories))

print(new_video_mp4 + " is categorized as " + mode(categories))
