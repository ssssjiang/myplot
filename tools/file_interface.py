import binascii
import csv
import io
import json
import logging
import os
import zipfile

import numpy as np

from myplot_tools import MyException
import myplot_tools.core.lie_algebra as lie
import myplot_tools.core.transformations as tr
from myplot_tools.core.trajectory import PosePath3D, PoseTrajectory3D
from myplot_tools.tools import user

logger = logging.getLogger(__name__)

SUPPORTED_ROS_MSGS = {
    "geometry_msgs/PoseStamped", "geometry_msgs/PoseWithCovarianceStamped",
    "geometry_msgs/TransformStamped", "nav_msgs/Odometry"
}


class FileInterfaceException(MyException):
    pass


def has_utf8_bom(file_path):
    size_bytes = os.path.getsize(file_path)
    if size_bytes < 3:
        return False
    with open(file_path, 'rb') as f:
        return not int(binascii.hexlify(f.read(3)), 16) ^ 0xEFBBBF


def csv_read_matrix(file_path, delim=',', comment_str="#"):
    """
    :param file_path:
    :param delim:
    :param comment_str:
    :return:
    """
    if hasattr(file_path, 'read'):
        generator = (line for line in file_path
                     if not line.startswith(comment_str))
        reader = csv.reader(generator, delimiter=delim)
        mat = [row for row in reader]
    else:
        if not os.path.isfile(file_path):
            raise FileInterfaceException('csv file ' + str(file_path) + " does not exist.")
        skip_3_bytes = has_utf8_bom(file_path)
        with open(file_path) as f:
            if skip_3_bytes:
                f.seek(3)
            generator = (line for line in f
                         if not line.startswith(comment_str))
            reader = csv.reader(generator, delimiter=delim)
            mat = [row for row in reader]
    return mat


def read_tum_trajectory_file(file_path):
    """
    :param file_path: file_path: the trajectory file path (or file handle)
    :return: trajectory.PoseTrajectory3D object
    """
    raw_mat = csv_read_matrix(file_path, delim=" ", comment_str="#")
    error_msg = ("TUM trajectory files must have 8 entries per row "
                 "and no trailing delimiter at the end of the rows (space)")

    #TODO: It doesn’t have to be 8
    if len(raw_mat) > 0 and len(raw_mat[0]) != 8:
        raise FileInterfaceException(error_msg)
    try:
        mat = np.array(raw_mat).astype(float)
    except ValueError:
        raise FileInterfaceException(error_msg)

    stamps = mat[:, 0]
    xyz = mat[:, 1: 4]
    quat = mat[:, 4:]
    quat = np.roll(quat, 1, axis=1)
    if not hasattr(file_path, 'read'):
        logger.debug("Loaded {} stamps and poses from: {}".format(
            len(stamps), file_path))
    return