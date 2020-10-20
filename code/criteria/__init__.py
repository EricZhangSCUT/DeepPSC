# -*- coding: utf-8 -*
import numpy as np
from .backbonerebuild import BackboneRebuild
from .criteria import cal_RMSD, cal_GDT, cal_rama


bbrebuild = BackboneRebuild().rebuild


def group_atoms(coo, cb):
    n = coo[3::4]
    c = coo[1::4]
    o = coo[2::4]
    min_bb = np.concatenate([n, c])
    ext_bb = np.concatenate([n, c, o, cb])
    return n, c, o, cb, min_bb, ext_bb


def strip_tgt(coo, cb, seq):
    if seq[0] != 'G':
        cb = cb[1:]
    if seq[-1] != 'G':
        cb = cb[:-1]
    return coo[1:-2], cb


def cal_criteria(coo, cb, coo_, cb_, seq, rmsd_norm=100, gdt_cutoffs=[0.05, 0.1, 0.2], rama_reduce_output=True):
    atoms_ = group_atoms(coo_, cb_)
    coo_striped, cb_striped = strip_tgt(coo, cb, seq)
    atoms = group_atoms(coo_striped, cb_striped)

    rmsd = cal_RMSD(atoms, atoms_, rmsd_norm)
    gdt = cal_GDT(atoms, atoms_, gdt_cutoffs)
    rama = cal_rama(coo_, seq, rama_reduce_output)
    return rmsd, gdt, rama
