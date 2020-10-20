import os
import numpy as np
from torch.utils.data import Dataset
import random


def fuse_groups(groups, target_size):
    groups.sort(key=lambda x: sum(x), reverse=True)
    for g in groups:
        g.sort(reverse=True)

    i = 0
    while True:
        if sum(groups[i])+groups[i+1][-1] <= target_size:
            groups[i].append(groups[i+1].pop(-1))
            if len(groups[i+1]) == 0:
                groups.pop(i+1)
                if i == len(groups)-1:
                    return groups
        else:
            if i+1 < len(groups)-1:
                i += 1
            else:
                return groups


def random_group(lengths, target_size):
    random.shuffle(lengths)
    groups = [[]]
    for l_ in lengths:
        if sum(groups[-1])+l_ <= target_size:
            groups[-1] += [l_]
        else:
            groups.append([l_])

    if len(groups) == 1:
        return groups

    groups_num = []
    groups_num.append(len(groups))
    while True:
        if len(groups_num) > 5:
            if groups_num[-1] == groups_num[-6]:
                break
        groups = fuse_groups(groups, target_size)
        groups_num.append(len(groups))
    return groups


def fill_groups(groups, len_sep_files):
    file_groups = []
    for g in groups:
        g.sort(reverse=True)
        file_groups.append([len_sep_files[l_].pop(0) for l_ in g])
    return file_groups


def group_files(len_sep_files, lengths, target_size):
    len_groups = random_group(lengths, target_size)
    grouped_files = fill_groups(len_groups, len_sep_files)
    return grouped_files


def load_compressed_array(filename):
    shape, keys, values = np.load(filename, allow_pickle=True)
    ary = np.zeros(shape)
    ary[keys] = values
    return ary.astype('float32')


class Image2Tor(Dataset):
    def __init__(self, dataset, with_target=True, file_list=None, aa_batchsize=800):
        self.input_path = './dataset/data/%s/image' % dataset

        self.with_target = with_target
        if with_target:
            self.target_path = './dataset/data/%s/tor' % dataset
            with open('./dataset/data/%s/dataset_list/%s.txt' % (dataset, file_list)) as f:
                file_list = f.read().split('\n')
        else:
            file_list = [filename[:-4]
                         for filename in os.listdir(self.input_path)]

        length = {}
        with open('./dataset/data/%s/len.txt' % dataset) as f:
            lines = f.read().split('\n')
            for line in lines:
                filename, l_ = line.split(':')
                length[filename] = int(l_)

        len_sep_files = [[] for _ in range(aa_batchsize+1)]
        lengths = []
        for filename in file_list:
            lengths.append(length[filename])
            len_sep_files[length[filename]].append(filename)

        self.batchs = group_files(len_sep_files, lengths, aa_batchsize)

    def __len__(self):
        return len(self.batchs)

    def __getitem__(self, idx):
        filenames = self.batchs[idx]
        images = []
        sincos = []
        lengths = []
        for filename in filenames:
            image = load_compressed_array(
                os.path.join(self.input_path, '%s.npy' % filename))
            image = np.swapaxes(image, 1, 3)
            images.append(image)
            if self.with_target:
                sincos.append(np.load(os.path.join(
                    self.target_path, '%s.npy' % filename)))
            lengths.append(np.shape(images[-1])[0])

        images = np.concatenate(images)
        if self.with_target:
            sincos = np.concatenate(sincos, 1)
        return images, sincos, filenames, lengths