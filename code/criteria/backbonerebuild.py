import numpy as np
import geo


class BackboneRebuild(object):
    def __init__(self):
        self.proj_C = np.array([0.8027563, 1.4310328])  # [cis, trans]
        self.radius_C = np.array([1.2996129, 0.527541])
        self.proj_N = np.array([0.8152914, 1.4068587])
        self.radius_N = np.array([1.2145333, 0.39022177])
        self.proj_O = np.array([0.22724794, 1.6512747])
        self.radius_O = np.array([2.3761292, 1.7392269])
        self.r_CB = 1.53534496
        self.t_CB = 0.91366992

    def sincos2tor(self, sincos):
        tor = np.array((np.arctan2(sincos[0], sincos[1]), np.arctan2(
            sincos[2], sincos[3]))).swapaxes(0, 1)
        return tor

    def tor2coo(self, tor, ca):
        ca_ca = (ca[1:]-ca[:-1])
        l_ca_ca = geo.get_len(ca_ca)
        is_tran = (l_ca_ca//3.4).astype(int).reshape(-1, 1)

        ori_ca_ca = geo.norm(ca_ca)
        cos_3ca = geo.batch_cos(ca_ca[:-1], ca_ca[1:])
        projection_ground = (
            l_ca_ca[1:]*cos_3ca).reshape(-1, 1)*geo.norm(ca_ca[:-1])
        last_projection_ground = l_ca_ca[-2]*cos_3ca[-1]*geo.norm(ca_ca[-1])
        ori_ground = np.concatenate(
            (geo.norm(ca_ca[1:]-projection_ground), geo.norm(last_projection_ground-ca_ca[-2])))

        ori_C = geo.rotation(ori_ground, ori_ca_ca, tor[:, 0].reshape(-1, 1))
        ori_N = geo.rotation(ori_ground, ori_ca_ca, tor[:, 1].reshape(-1, 1))

        C = ori_C * self.radius_C[is_tran] + \
            ori_ca_ca*self.proj_C[is_tran] + ca[:-1]
        O = ori_C * self.radius_O[is_tran] + \
            ori_ca_ca*self.proj_O[is_tran] + ca[:-1]
        N = ori_N * self.radius_N[is_tran] - \
            ori_ca_ca*self.proj_N[is_tran] + ca[1:]

        coo = np.concatenate([ca[np.newaxis, :-1], [C, O, N]]
                             ).swapaxes(0, 1).reshape(-1, 3)
        coo = np.concatenate((coo, ca[np.newaxis, -1]))
        return coo

    def coo2cb(self, coo, seq):
        mask = np.array([aa != 'G' for aa in seq]).nonzero()
        cb = []
        n = coo[::4][mask]
        ca = coo[1::4][mask]
        c = coo[2::4][mask]
        ori_ca_n = geo.norm(n - ca)
        ori_ca_c = geo.norm(c - ca)
        ori_mid = geo.norm(ori_ca_n + ori_ca_c)
        rot_axis_cb = geo.norm(ori_ca_c - ori_ca_n)
        cb = geo.rotation(ori_mid, rot_axis_cb, self.t_CB) * self.r_CB + ca
        return np.array(cb)

    def rebuild(self, ca, tor, seq=None, from_sincos=True, with_cb=True):
        if from_sincos:
            coo = self.tor2coo(self.sincos2tor(tor), ca).astype('float32')
        else:
            coo = self.tor2coo(ca).astype('float32')

        if with_cb:
            cb = self.coo2cb(coo[3:-1], seq[1:-1]).astype('float32')
            return coo, cb
        else:
            return coo
