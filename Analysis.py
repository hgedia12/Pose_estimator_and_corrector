import pandas as pd
import matplotlib.pyplot as plt
import cv2
import json
import os
import numpy as np


name='correct1'
filename = 'Resources/Aasan1_with_names/'+name+'mp4'
path_to_json = "Resources/Aasan1output2/"+name+"/"

# Load key point data from JSON output
column_names = ['x', 'y', 'acc']


# Import Json files, pos_json = position JSON
json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]
print('Found: ', len(json_files), 'json keypoint frame files')
count = 0

# Video Dimensions
cap=cv2.VideoCapture('Resources/Aasan1_with_names'+name+'.mp4')
width=cap.get(3)
height=cap.get(4)

# instanciate data frames
body_keypoints_df = pd.DataFrame()
print('json files: ', json_files[0])

# Loop through all json files in output directory
# Each file is a frame in the video
# If multiple people are detected - choose the most centered high confidence points
for file in json_files:

    temp_df = json.load(open(path_to_json + file))
    temp = []
    n=0
    for k, v in temp_df['part_candidates'][0].items():
        # Single point detected
        if len(v) < 4:
            temp.append(v)
            # print('Extracted highest confidence points: ',v)

        # Multiple points detected
        elif len(v) > 4:
            near_middle = width
            np_v = np.array(v)

            # Reshape to x,y,confidence
            np_v_reshape = np_v.reshape(int(len(np_v) / 3), 3)
            np_v_temp = []
            # compare x values
            for pt in np_v_reshape:
                if (np.absolute(pt[0] - width / 2) < near_middle):
                    near_middle = np.absolute(pt[0] - width / 2)
                    np_v_temp = list(pt)

            temp.append(np_v_temp)
            # print('Extracted highest confidence points: ',v[index_highest_confidence-2:index_highest_confidence+1])
        else:
            # No detection - record zeros
            temp.append([0, 0, 0])

    temp_df = pd.DataFrame(temp)
    temp_df = temp_df.fillna(0)
    #print(temp_df)

    try:
        prev_temp_df = temp_df
        body_keypoints_df = body_keypoints_df.append(temp_df)

    except:
        print('please shoot the video again: ', file)

body_keypoints_df.columns = column_names
body_keypoints_df.reset_index(drop=True)

print('length of merged keypoint set: ',body_keypoints_df.size)
n=0

cap=cv2.VideoCapture('Resources/Aasan1_with_names'+name+'.mp4')

'''while True:
    success,img = cap.read()
    print(img)
    if success:
        x,y=body_keypoints_df.iloc[25*n+9,0:2]
        cv2.circle(img,(int(x),int(y)), 10,(0,0,0),2)
        x,y=body_keypoints_df.iloc[25*n+10,0:2]
        cv2.circle(img,(int(x),int(y)), 10,(0,0,0),2)
        cv2.imshow('hi',img)
        cv2.waitKey(10)
        n=n+1
    else:
        print("finished")
        break'''

main_df_mean=pd.read_csv('Resources/Outputcsv/correct4.csv', index_col=0)
main_df_median=pd.read_csv('Resources/Outputcsv/correct4.csv', index_col=0)
main_df2_mean=pd.read_csv('Resources/Outputcsv/incorrect8.csv', index_col=0)
main_df2_median=pd.read_csv('Resources/Outputcsv/incorrect8.csv', index_col=0)

for i in range(0, len(main_df_median.index) - 30):
    for j in main_df_median.columns:
        main_df_median.iloc[i][j] = main_df_mean.iloc[i:i + 20][j].median()

for i in range(0, len(main_df2_median.index) - 30):
    for j in main_df_median.columns:
        main_df2_median.iloc[i][j] = main_df2_mean.iloc[i:i + 20][j].median()

for i in range(0,len(main_df2_median.index)-30):
    main_df_mean.iloc[i]=main_df_mean.iloc[i:i+30].mean()

for i in range(0, len(main_df2_mean.index) - 30):
    main_df2_mean.iloc[i]=main_df2_mean.iloc[i:i+30].mean()



print(main_df_mean.columns)
plt.subplot(3,3,1)
plt.plot(main_df_median['lshoulderangle'])
plt.plot(main_df2_median['lshoulderangle'])

plt.subplot(3,3,2)
plt.plot(main_df_median['lelbowangle'])
plt.plot(main_df2_median['lelbowangle'])

plt.subplot(3,3,3)
plt.plot(main_df_median['lhipangle'])
plt.plot(main_df2_median['lhipangle'])

plt.subplot(3,3,4)
plt.plot(main_df_median['lkneeangle'])
plt.plot(main_df2_median['lkneeangle'])

plt.subplot(3,3,5)
plt.plot(main_df_median['lelbowratio'])
plt.plot(main_df2_median['lelbowratio'])

plt.subplot(3,3,6)
plt.plot(main_df_median['lshoulderratio'])
plt.plot(main_df2_median['lshoulderratio'])

plt.subplot(3,3,7)
plt.plot(main_df_mean['lshoulderratio'])
plt.plot(main_df2_mean['lshoulderratio'])



plt.show()

