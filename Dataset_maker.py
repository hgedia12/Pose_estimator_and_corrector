import numpy as np
import os
import cv2
import json
import math
import matplotlib.pyplot as plt
import pandas as pd
from plotly import tools
import chart_studio.plotly as py
import plotly.graph_objs as go

print('OpenCV - version: ', cv2.__version__)



def getAngle(a, b, c):
    ang = math.degrees(math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0]))
    num=math.sqrt( ((a[0]-b[0])**2)+((a[1]-b[1])**2))
    denum=math.sqrt( ((b[0]-c[0])**2)+((b[1]-c[1])**2))
    if denum==0:
        return [ang+360,0] if ang < 0 else [ang,0]
    else:
        rat=num/denum
        return [ang + 360,rat] if ang < 0 else [ang,rat]


# write name
name='correct6'
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
result=[]
while n<len(json_files):
    x=[]
    y=[]
    for i in range(0,25):
        x.append(int(body_keypoints_df.iloc[25*n+i,0]))
        y.append(int(body_keypoints_df.iloc[25*n+i,1]))

    lhipangle,lhipratio=getAngle((x[13],y[13]),(x[12],y[12]),(x[9],y[9]))
    rkneeangle,rkneeratio=(getAngle((x[11],y[11]),(x[10],y[10]),(x[9],y[9])))
    rhipangle,rhipratio=(getAngle((x[10],y[10]),(x[9],y[9]),(x[12],y[12])))
    lkneeangle,lkneeratio=(getAngle((x[14],y[14]),(x[13],y[13]),(x[12],y[12])))
    rshoulderangle,rshoulderratio=(getAngle((x[3],y[3]),(x[2],y[2]),(x[9],y[9])))
    lshoulderangle,lshoulderratio=(getAngle((x[6],y[6]),(x[5],y[5]),(x[12],y[12])))
    lelbowangle,lelbowratio=(getAngle((x[7],y[7]),(x[6],y[6]),(x[5],y[5])))
    relbowangle,relbowratio=(getAngle((x[4],y[4]),(x[3],y[3]),(x[2],y[2])))

    new_list=[lhipangle, lhipratio, rhipangle, rhipratio, lkneeangle, lkneeratio, rkneeangle, rkneeratio, lshoulderangle,
        lshoulderratio, rshoulderangle, rshoulderratio,lelbowangle,lelbowratio,relbowangle,relbowratio]

    result.append(new_list)
    n=n+1



result_df=pd.DataFrame(result)
result_df.columns=['lhipangle', 'lhipratio', 'rhipangle', 'rhipratio', 'lkneeangle', 'lkneeratio', 'rkneeangle', 'rkneeratio', 'lshoulderangle',
         'lshoulderratio', 'rshoulderangle', 'rshoulderratio','lelbowangle','lelbowratio','relbowangle','relbowratio']

result_df.to_csv('Resources/Outputcsv/'+name+'.csv')

