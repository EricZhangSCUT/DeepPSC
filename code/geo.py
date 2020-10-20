# -*- coding: utf-8 -*
import numpy as np


def rotation(vec, axis, theta_):
    theta = theta_ + np.pi
    cos = np.cos(theta)
    vec_rot = cos*vec + \
        np.sin(theta)*np.cross(axis, vec) + \
        (1 - cos) * batch_dot(vec, axis).reshape(-1, 1) * axis
    if len(vec_rot) == 1:
        vec_rot = vec_rot[0]
    return vec_rot


def batch_dot(vecs1, vecs2):
    if len(vecs2.shape) == 1:
        vecs2 = vecs2.reshape(1, -1)
    return np.diag(np.matmul(vecs1, vecs2.T))


def get_torsion(vec1, vec2, axis):
    n = np.cross(axis, vec2)
    n2 = np.cross(vec1, axis)
    sign = np.sign(batch_cos(vec1, n))
    angle = np.arccos(batch_cos(n2, n))
    torsion = sign*angle
    if len(torsion) == 1:
        return torsion[0]
    else:
        return torsion


def get_len(vec):
    return np.linalg.norm(vec, axis=-1)


def norm(vec):
    return vec / get_len(vec).reshape(-1, 1)


def get_angle(vec1, vec2):
    return np.arccos(np.dot(norm(vec1), norm(vec2)))


def batch_cos(vecs1, vecs2):
    cos = np.diag(np.matmul(norm(vecs1), norm(vecs2).T))
    cos = np.clip(cos, -1, 1)
    return cos
