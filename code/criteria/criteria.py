import geo
import numpy as np
import os


def cRMSD(coo1, coo2, norm=100):
    rmsd = np.sqrt(np.mean(np.square(np.linalg.norm(coo1-coo2, axis=-1))))

    if norm:
        norm_rmsd = rmsd / (1 + np.log(np.sqrt(len(coo1) / norm)))
        return norm_rmsd
    else:
        return rmsd


def GDT(coo1, coo2, cutoff):
    distance = geo.get_len(coo1-coo2)
    count = np.sum(distance <= cutoff)
    return count/len(coo1)


def cal_GDT(coo1, coo2, cutoffs):
    gdt = [[GDT(tgt, out, cutoff) for cutoff in cutoffs]
           for tgt, out in zip(coo1, coo2)]
    return np.array(gdt).astype('float32')


def cal_RMSD(coo1, coo2, norm):
    rmsd = [cRMSD(tgt, out, norm) for tgt, out in zip(coo1, coo2)]
    return np.array(rmsd).astype('float32')


RAMA_SETTING = {
    "General": {
        "file": os.path.join('./criteria/rama_contour', 'pref_general.data'),
        "bounds": [0, 0.0005, 0.02, 1]
    },
    "GLY": {
        "file": os.path.join('./criteria/rama_contour', 'pref_glycine.data'),
        "bounds": [0, 0.002, 0.02, 1]
    },
    "PRO": {
        "file": os.path.join('./criteria/rama_contour', 'pref_proline.data'),
        "bounds": [0, 0.002, 0.02, 1]
    },
    "PRE-PRO": {
        "file": os.path.join('./criteria/rama_contour', 'pref_preproline.data'),
        "bounds": [0, 0.002, 0.02, 1]
    }
}


def load_rama_map(filename):
    rama_map = np.zeros((360, 360), dtype=np.float64)
    with open(filename) as fn:
        for line in fn:
            if line.startswith("#"):
                continue
            else:
                line = line.split()
                x = int(float(line[1]))
                y = int(float(line[0]))
                rama_map[x + 180][y + 180] = \
                    rama_map[x + 179][y + 179] = \
                    rama_map[x + 179][y + 180] = \
                    rama_map[x + 180][y + 179] = float(line[2])
    return rama_map


for rama_type in RAMA_SETTING.keys():
    RAMA_SETTING[rama_type]['map'] = load_rama_map(
        RAMA_SETTING[rama_type]['file'])


def cal_phipsi(coo):
    ca = coo[::4][1:-1]
    n = coo[3::4]
    c = coo[1::4]

    c_n = geo.norm(n-c)
    n_ca = geo.norm(ca-n[:-1])
    ca_c = geo.norm(c[1:] - ca)

    phi = geo.get_torsion(c_n[:-1], ca_c, n_ca) / np.pi * 180 + 180
    psi = geo.get_torsion(n_ca, c_n[1:], ca_c) / np.pi * 180 + 180

    phi[phi >= 360] -= 360
    psi[psi >= 360] -= 360
    return phi, psi


def seq2rama_type(seq):
    rama_types = []
    for aa in seq:
        if aa == 'G':
            rama_types.append('GLY')
        elif aa == 'P':
            rama_types.append('PRO')
            if len(rama_types) != 1:
                if rama_types[-2] != 'PRO':
                    rama_types[-2] = "PRE-PRO"
        else:
            rama_types.append('General')
    return rama_types


def cal_rama(coo, seq, reduce_output=True):
    rama_types = seq2rama_type(seq)[1:-1]
    phis, psis = cal_phipsi(coo)
    core = {}
    allow = {}
    outlier = {}

    for rank in core, allow, outlier:
        for rama_type in RAMA_SETTING.keys():
            rank[rama_type] = {}

    for index, (phi, psi, rama_type) in enumerate(zip(phis, psis, rama_types)):
        if RAMA_SETTING[rama_type]['map'][int(psi)][int(phi)] < RAMA_SETTING[rama_type]["bounds"][1]:
            outlier[rama_type][index] = [psi, phi]
        elif RAMA_SETTING[rama_type]['map'][int(psi)][int(phi)] < RAMA_SETTING[rama_type]["bounds"][2]:
            allow[rama_type][index] = [psi, phi]
        else:
            core[rama_type][index] = [psi, phi]

    core_num = [len(core[rama_type].keys())
                for rama_type in RAMA_SETTING.keys()]
    allow_num = [len(allow[rama_type].keys())
                 for rama_type in RAMA_SETTING.keys()]
    outlier_num = [len(outlier[rama_type].keys())
                   for rama_type in RAMA_SETTING.keys()]

    core_num = np.array(core_num + [sum(core_num)])
    allow_num = np.array(allow_num + [sum(allow_num)])
    outlier_num = np.array(outlier_num + [sum(outlier_num)])
    total_num = core_num + allow_num + outlier_num

    rama_matrix = np.concatenate(
        (core_num, allow_num, outlier_num, total_num)).reshape(4, -1)

    core_rate, allow_rate, outlier_rate = rama_matrix[:3, -1] / \
        rama_matrix[-1, -1]

    if reduce_output:
        return [core_rate, allow_rate, outlier_rate]
    else:
        return rama_matrix, [core, allow, outlier]
