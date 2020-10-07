import myplot_tools.colmap_helper.model_utils.read_write_model as rw
import myplot_tools.colmap_helper.visualization as vis
import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib import cm
import os
from scipy.spatial.transform import Rotation as R, Slerp
from pathlib import Path



vertices_file = "/home/songshu/dataset/relocalization/2019-09-26_11-47-37__test/d99ed3c3ffffffff2100000000000000/vertices.csv"
imgTime = "/home/songshu/dataset/relocalization/2019-09-26_11-47-37__test/fisheye_timestamps.txt"

tmp = np.loadtxt(vertices_file, dtype=np.str, delimiter=",")
tmp = tmp[1:]
viMap=[]


for frame in tmp:
    viMap.append([float(a) for a in frame])

for i in range(len(viMap)):
    p = viMap[i]
    T_w_c = np.eye(4)
    T_w_c[:3, 3] = np.array([p[2], p[3], p[4]])
    T_w_c[:3, :3] = (R.from_quat([p[5], p[6], p[7], p[8]])).as_dcm()
    print(T_w_c)
    print()
    trans = np.eye(4)
    T_w_c = np.dot(trans, T_w_c)
    print(T_w_c)
    print()
    print("============================================")
    T_c_w = np.linalg.inv(T_w_c)