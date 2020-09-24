from __future__ import print_function  # Python 2.7 backwards compatibility

import os
import logging
import pickle
import collections
from enum import Enum

import matplotlib as mpl
from myplot_tools.tools.settings import SETTINGS

mpl.use(SETTINGS.plot_backend)

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.art3d as art3d
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import LineCollection
from matplotlib.transforms import Affine2D

import numpy as np
import seaborn as sns

from myplot_tools import MyException
from myplot_tools.tools import user
from myplot_tools.core import trajectory
