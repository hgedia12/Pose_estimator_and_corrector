import numpy as np
import math
import glob
import utils
import pandas as pd
from pprint import pprint
from scipy.signal import medfilt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import DistanceMetric
from sklearn.metrics import classification_report

def dtw(s1, s2):
    DTW={}

    for i in range(len(s1)):
        DTW[(i, -1)] = float('inf')
    for i in range(len(s2)):
        DTW[(-1, i)] = float('inf')
    DTW[(-1, -1)] = 0

    for i in range(len(s1)):
        for j in range(len(s2)):
            dist= (s1[i]-s2[j])**2
            DTW[(i, j)] = dist + min(DTW[(i-1, j)],DTW[(i, j-1)], DTW[(i-1, j-1)])

    return np.sqrt(DTW[len(s1)-1, len(s2)-1])

main_df_mean=pd.read_csv('Resources/Outputcsv/correct4.csv', index_col=0)
main_df_median=pd.read_csv('Resources/Outputcsv/correct4.csv', index_col=0)
main_df2_mean=pd.read_csv('Resources/Outputcsv/incorrect7.csv', index_col=0)
main_df2_median=pd.read_csv('Resources/Outputcsv/incorrect7.csv', index_col=0)

for i in range(0,len(main_df_median.index)-30):
    for j in main_df_median.columns:
        main_df_median.iloc[i][j]=main_df_mean.iloc[i:i+30][j].median()

for i in range(0,len(main_df2_median.index)-30):
    for j in main_df_median.columns:
        main_df2_median.iloc[i][j]=main_df2_mean.iloc[i:i+30][j].median()


print(dtw(list(main_df_median['lhipangle']),list(main_df2_median['lhipangle'])))