import copy
import logging

import numpy as np

from myplot_tools import MyException
import myplot_tools.core.transformations as tr
import myplot_tools.core.geometry as geometry
from myplot_tools.core import lie_algebra as lie