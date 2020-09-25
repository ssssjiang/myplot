import copy
import logging

import numpy as np

from myplot_tools import MyException
import myplot_tools.core.transformations as tr
import myplot_tools.core.geometry as geometry
from myplot_tools.core import lie_algebra as lie

logger = logging.getLogger(__name__)


class TrajectoryException(MyException):
    pass


class PosePath3D(object):
    """
    just a path, no temporal information
    also:base class for real trajectory
    """
    def __init__(self, positions_xyz=None, orientations_quat_wxyz=None,
                 poses_se3=None, meta=None):
        """
        :param positions_xyz: nx3 list of x, y, z position.
        :param orientations_quat_wxyz: nx4 list of quaternions (w, x, y, z format)
        :param poses_se3: list of SE(3) poses
        :param meta: optional metadata
        """
        # initial failure.
        if (positions_xyz is None or orientations_quat_wxyz is None) and poses_se3 is None:
            raise TrajectoryException("must provide at least positions_xyz "
                                      "& orientations_quat_wxyz or poses_se3")
        if positions_xyz is not None:
            self._position_xyz = np.array(positions_xyz)
        if orientations_quat_wxyz is not None:
            self._orientations_quat_wxyz = np.array(orientations_quat_wxyz)
        if poses_se3 is not None:
            self._poses_se3 = poses_se3
        self.meta = {} if meta is None else meta

    def __str__(self):
        return "{} poses, {:, .3f}m path length".format(self.num_poses, self.path_length)

    def __eq__(self, other):
        if type(other) != type(self):
            return False
        if not self.num_poses == other.num_poses:
            return False
        equal = True
        equal &= all([
            np.allclose(p1, p2)
            for p1, p2 in zip(self._poses_se3, other._poses_se3)
        ])
        equal &= np.allclose(self.orientations_quat_wxyz)


    @property
    def positions_xyz(self):
        if not hasattr(self, "_positions_xyz"):
            assert hasattr(self, "_poses_se3")
            self._position_xyz = np.array([p[:3, 3] for p in self._poses_se3])
        return self._position_xyz

    @property
    def distances(self):
        return geometry.ac

    @property
    def num_poses(self):
        if hasattr(self, "_poses_se3"):
            return len(self._poses_se3)
        else:
            return self.position_xyz.shape[0]


class PoseTrajectory3D(PosePath3D, object):


def calc_speed(xyz_1, xyz_2, t_1, t_2):
    """
    :param xyz_1: position at timestamp 1
    :param xyz_2: position at timestamp 2
    :param t_1: timestamp 1
    :param t_2: timestamp 2
    :return: speed in m/s
    """
    if (t_2 - t_1) <= 0:
        raise TrajectoryException("bad timestamp: " + str(t_1) + " & " + str(t_2))
    return np.linalg.norm(xyz_2 - xyz_1) / (t_2 - t_1)


def calc_angle_speed(p_1, p_2, t_1, t_2, degrees=False):
    """
    :param p_1: pose at timestamp 1
    :param p_2: pose at timestamp 2
    :param t_1: timestamp 1
    :param t_2: timestamp 2
    :param degrees: set to True to return deg/s
    :return: speed in rad/s
    """
    if (t_2 - t_1) <= 0:
        raise TrajectoryException("bad timestamp: " + str(t_1) + " & " + str(t_2))

    if degrees:
        angle_1 = lie.so3_log(p_1[:3, :3]) * 180 / np.pi
        angle_2 = lie.so3_log(p_2[:3, :3]) * 180 / np.pi
    else:
        angle_1 = lie.so3_log(p_1[:3, :3])
        angle_2 = lie.so3_log(p_2[:3, :3])
    return (angle_2 - angle_1) / (t_2 - t_1)


def xyz_quat_wxyz_to_se3_poses(xyz, quat):
    """
    :param xyz: position
    :param quat: pose
    :return: se3 pose
    """
    poses = [
        lie.se3(lie.so3_from_se3(tr.quaternion_matrix(quat)), xyz)
        for quat, xyz in zip(quat, xyz)
    ]
    return poses


def se3_poses_to_xyz_quat_wxyz(poses):
    xyz = np.array([pose[:3, 3] for pose in poses])
    quat_wxyz = np.array([tr.quaternion_from_matrix(pose) for pose in poses])
    return xyz, quat_wxyz


def align_trajectory(traj, traj_ref, correct_scale=False,
                     correct_only_scale=False, n=-1, return_parameters=False):
    """
    :param traj: the trajectory to align
    :param traj_ref: reference trajectory
    :param correct_scale: set to True to adjust also the scale
    :param correct_only_scale: set to True to correct the scale, but not the pose
    :param n: the number of poses to use, counted from the start (default: all)
    :param return_parameters: also return result parameters of Umeyama's method
    :return: the aligned trajectory.
     If return_parameters is set, the rotation matrix, translation vector and
    scaling parameter of Umeyama's method are also returned.
    """

    traj_aligned = copy.deepcopy(traj)
    with_scale = correct_scale or correct_only_scale
    if correct_only_scale:
        logger.debug("Correcting scale...")
    else:
        logger.debug("Aligning using Umeyama's method..." +
                     (" (with scale correction)" if with_scale else ""))
    if n == -1:
        r_a, t_a, s = geometry.umeyama_alignment(traj_aligned.positions_xyz.T,
                                                 traj_ref.positions_xyz.T,
                                                 with_scale)
    else:
        r_a, t_a, s = geometry.umeyama_alignment(traj_aligned.positions_xyz[:n, :].T,
                                                 traj_ref.positions_xyz[:n, :].T,
                                                 with_scale)

    if not correct_only_scale:
        logger.debug("Rotation of alignment:\n{}"
                     "\nTranslation of alignment:\n{}".format(r_a, t_a))
    logger.debug("Scale correction； {}".format(s))

    if correct_only_scale:
        traj_aligned.scale(s)
    elif correct_scale:
        traj_aligned.scale(s)
        traj_aligned.transform(lie.se3(r_a, t_a))
    else:
        traj_aligned.transform(lie.se3(r_a, t_a))

    if return_parameters:
        return traj_aligned, r_a, t_a, s
    else:
        return traj_aligned






