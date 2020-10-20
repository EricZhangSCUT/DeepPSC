# -*- coding: utf-8 -*
import numpy as np
import geo
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from aa_encoder import AminoacidEncoder
import pathlib
import os


def MapDis(coo):
    return squareform(pdist(coo, metric='euclidean')).astype('float32')


class Atom(object):
    def __init__(self, aminoacid, index, x, y, z):
        self.aa = aminoacid
        self.index = index
        self.x = x
        self.y = y
        self.z = z


class Arraylize(object):
    def __init__(self, resolution, size, atoms, indexs, aa_encoder):
        self.atoms = atoms
        self.pad = 4
        self.ar = size + self.pad
        self.idx_ary = indexs
        self.scale = size * 2 / resolution
        self.res = resolution + int(2*self.pad/self.scale)
        self.dim = 5
        self.array = np.zeros(
            [self.res, self.res, self.dim], dtype='float32', order='C')
        self.aa_encoder = aa_encoder
        self.rec = {}
        self.site = {}
        self.run()

    def pixel_center_dis(self, dot):
        dot.dis_x = dot.x / self.scale % 1 - 0.5
        dot.dis_y = dot.y / self.scale % 1 - 0.5
        dot.dis_sqrt = dot.dis_x ** 2 + dot.dis_y ** 2

    def closer_pixel(self, dot):
        x_sign = int(np.sign(dot.dis_x))
        y_sign = int(np.sign(dot.dis_y))
        if abs(dot.dis_x) < abs(dot.dis_y):
            neighbors = [(0, y_sign), (x_sign, 0), (x_sign, y_sign), (-x_sign, 0),
                         (-x_sign, y_sign), (0, -y_sign), (x_sign, -y_sign), (-x_sign, -y_sign)]
        else:
            neighbors = [(x_sign, 0), (0, y_sign), (x_sign, y_sign), (0, -y_sign),
                         (x_sign, -y_sign), (-x_sign, 0), (-x_sign, y_sign), (-x_sign, -y_sign)]
        for (i, j) in neighbors:
            if -1 < dot.x_ary + i < self.res and -1 < dot.y_ary + j < self.res:
                if self.array[dot.x_ary + i, dot.y_ary + j, -1] == 0:
                    dot.x_ary = dot.x_ary + i
                    dot.y_ary = dot.y_ary + j
                    self.draw_atom(dot)
                    break

    def closer_dot(self, dot1, dot2):
        self.pixel_center_dis(dot1)
        self.pixel_center_dis(dot2)
        if dot1.dis_sqrt > dot2.dis_sqrt:
            self.closer_pixel(dot1)
            self.draw_atom(dot2)
        else:
            self.closer_pixel(dot2)

    def draw_atom(self, dot):
        self.array[dot.x_ary, dot.y_ary] = [
            dot.z, dot.index] + list(self.aa_encoder.encode(dot.aa)[0])
        self.rec.update({(dot.x_ary, dot.y_ary): dot})

    def draw_dot(self, x, y, dot, z_add, idx_add, property_add):
        if self.rec.get((x, y)) is None:
            property_inter = list(
                self.aa_encoder.encode(dot.aa)[0] + property_add)
            if self.array[x, y, 0]:
                if dot.z + z_add > self.array[x, y, 0]:
                    self.array[x, y] = [dot.z + z_add,
                                        dot.index + idx_add] + property_inter
            else:
                self.array[x, y] = [dot.z + z_add,
                                    dot.index + idx_add] + property_inter

    def dots_connection(self, dot1, dot2):
        z_dis = dot2.z - dot1.z
        x_sign = int(np.sign(dot2.x_ary - dot1.x_ary))
        y_sign = int(np.sign(dot2.y_ary - dot1.y_ary))
        x_dis = abs(dot2.x_ary - dot1.x_ary)
        y_dis = abs(dot2.y_ary - dot1.y_ary)
        long_step = max(x_dis, y_dis)
        short_step = min(x_dis, y_dis)
        property_dis = self.aa_encoder.encode(
            dot2.aa)[0]-self.aa_encoder.encode(dot1.aa)[0]

        if short_step == 0:
            if x_dis > y_dis:
                x_step, y_step = 1, 0
            else:
                x_step, y_step = 0, 1
        else:
            slope = long_step / short_step
            if x_dis > y_dis:
                x_step, y_step = 1, 1 / slope
            else:
                x_step, y_step = 1 / slope, 1

        for step in range(1, long_step):
            self.draw_dot(round(dot1.x_ary + step * x_step * x_sign), round(dot1.y_ary + step * y_step * y_sign),
                          dot1, z_dis * step / (long_step + 1), step / (long_step + 1), property_dis * step / (long_step + 1))

    def draw_connection(self):
        for (x, y) in self.rec.keys():
            self.site.update({self.rec[(x, y)]: [x, y]})
        for i in range(len(self.atoms) - 1):
            if self.atoms[i + 1].index - self.atoms[i].index == 1 or self.atoms[i].index == -1:
                self.dots_connection(self.atoms[i], self.atoms[i + 1])

    def crop_image(self):
        padding = int(self.pad / self.scale)
        self.array = self.array[padding:self.res -
                                padding, padding:self.res-padding]

    def height_limit(self):
        self.array[abs(self.array[:, :, 0]) > self.ar - self.pad] = 0

    def height_norm(self):
        self.array[:, :, 0] /= self.ar - self.pad

    def index_norm(self, norm_lenght=200):
        self.array[:, :, 1] /= norm_lenght

    def run(self):
        for atom in self.atoms:
            atom.x_ary = int(atom.x // self.scale + self.res // 2)
            atom.y_ary = int(atom.y // self.scale + self.res // 2)
            if self.rec.get((atom.x_ary, atom.y_ary)):
                self.closer_dot(self.rec[(atom.x_ary, atom.y_ary)], atom)
            else:
                self.draw_atom(atom)

        self.draw_connection()
        self.crop_image()
        self.height_limit()
        self.height_norm()
        self.index_norm()


class StrucRep(object):
    def __init__(self, struc_format='knn', aa_format='property', index_norm=200):
        self.aa_encoder = AminoacidEncoder(aa_format)
        self.index_norm = index_norm
        if struc_format == 'knn':
            self.struc_rep = self.knn_struc_rep
        elif struc_format == 'image':
            self.struc_rep = self.image_struc_rep
        elif struc_format == 'conmap':
            self.struc_rep = self.contact_map
        elif struc_format == 'dismap':
            self.struc_rep = self.distance_map

    def knn_struc_rep(self, ca, seq, k=15):
        dismap = MapDis(ca)
        nn_indexs = np.argsort(dismap, axis=1)[:, :k]
        relative_indexs = nn_indexs.reshape(-1, k, 1) - \
            nn_indexs[:, 0].reshape(-1, 1, 1).astype('float32')
        relative_indexs /= self.index_norm
        seq_embeded = self.aa_encoder.encode(seq)
        knn_feature = np.array(seq_embeded)[nn_indexs]
        knn_distance = [dismap[i][nn_indexs[i]] for i in range(len(nn_indexs))]
        knn_distance = np.array(knn_distance).reshape(-1, k, 1)

        knn_orient = []
        for i in range(len(nn_indexs)):
            orient = geo.norm(ca[nn_indexs[i]][1:] - ca[i])
            knn_orient.append(np.concatenate([np.zeros((1, 3)), orient]))
        knn_orient = np.array(knn_orient)

        knn_rep = np.concatenate(
            (knn_orient, knn_distance, relative_indexs, knn_feature), -1)
        return knn_rep.astype('float32')

    def contact_map(self, ca, seq='', cutoff=8):
        dismap = MapDis(ca)
        conmap = np.zeros_like(dismap)
        conmap[dismap < cutoff] = 1.
        return conmap.astype('float32')

    def distance_map(self, ca, seq=''):
        return MapDis(ca).astype('float32')

    def image_struc_rep(self, ca, seq, resolution=128, box_size=8, compress=True, pad=4):
        arrays = []
        tgt_x = np.array([0, 1, 0])
        rot_axis_y = tgt_x
        tgt_y = np.array([1, 0, 0])
        ori_x = geo.norm(ca[1:] - ca[:-1])
        ori_y = np.concatenate((ori_x[1:], -(ori_x[np.newaxis, -2])))

        centers = ca.copy()
        ori_x = np.concatenate((ori_x, ori_x[np.newaxis, -1]))
        ori_y = np.concatenate((ori_y, ori_y[np.newaxis, -1]))
        rot_axis_x = geo.norm(np.cross(ori_x, tgt_x))

        tor_x = geo.get_torsion(ori_x, tgt_x, rot_axis_x)
        ori_y_rot = geo.rotation(ori_y, rot_axis_x, tor_x.reshape(-1, 1))
        ori_y_proj = ori_y_rot.copy()
        ori_y_proj[:, 1] = 0.
        ori_y_proj = geo.norm(ori_y_proj)
        l_ori_y_proj = len(ori_y_proj)
        tor_y = geo.get_torsion(ori_y_proj,
                                np.tile(tgt_y, (l_ori_y_proj, 1)),
                                np.tile(rot_axis_y, (l_ori_y_proj, 1)))

        for i, center in enumerate(centers):
            ca_ = ca - center
            global_indexs = np.where(geo.get_len(
                ca_) < (box_size + pad)*np.sqrt(3))[0]
            local_indexs = global_indexs - i

            num_local_atoms = len(global_indexs)
            ca_xrot = geo.rotation(ca_[global_indexs],
                                   np.tile(rot_axis_x[i],
                                           (num_local_atoms, 1)),
                                   np.tile(tor_x[i], (num_local_atoms, 1)))
            ca_rot = geo.rotation(ca_xrot,
                                  np.tile(rot_axis_y, (num_local_atoms, 1)),
                                  np.tile(tor_y[i], (num_local_atoms, 1)))

            local_atoms = []
            for j, idx in enumerate(global_indexs):
                if np.max(np.abs(ca_rot[j])) < box_size + pad:
                    local_atoms.append(
                        Atom(seq[idx], local_indexs[j], ca_rot[j][0], ca_rot[j][1], ca_rot[j][2]))

            arrays.append(Arraylize(resolution=resolution,
                                    size=box_size,
                                    atoms=local_atoms,
                                    indexs=local_indexs,
                                    aa_encoder=self.aa_encoder).array)

        arrays = np.array(arrays, dtype='float32')

        if compress:
            shape = arrays.shape
            keys = arrays[:, :, :, -1].nonzero()
            values = arrays[keys]
            com_ary = [shape, keys, values.astype('float32')]
            return com_ary
        else:
            return arrays


if __name__ == "__main__":
    dataset = 'test'
    struc_format = 'dismap'
    coo_path = './data/%s/coo' % dataset
    seq_path = './data/%s/seq' % dataset

    strucrep = StrucRep(struc_format=struc_format)
    struc_path = './data/%s/%s' % (dataset, struc_format)
    pathlib.Path(struc_path).mkdir(parents=True, exist_ok=True)

    for filename in os.listdir(coo_path):
        coo = np.load(os.path.join(coo_path, filename))
        with open(os.path.join(seq_path, "%s.txt" % filename[:-4])) as f:
            seq = f.read()
        struc = strucrep.struc_rep(coo[1::4], seq)
        np.save(os.path.join(struc_path, filename), struc)
