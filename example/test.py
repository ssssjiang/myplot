import myplot_tools.colmap_helper.model_utils.read_write_model as rw
import myplot_tools.colmap_helper.visualization as vis
import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib import cm
import os
from scipy.spatial.transform import Rotation as R, Slerp
from pathlib import Path
import sqlite3

with open("test.txt", 'r') as f:
    data = f.read().splitlines()
    print(data)
    print(type(data))
