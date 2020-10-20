import torch
from torch.utils.data import DataLoader
import pathlib
import os
import numpy as np
import dataset
from criteria import cal_criteria, bbrebuild


def loss_from_log(train_name):
    with open('../logs/log_%s.txt' % train_name) as f:
        lines = f.readlines()
    val_loss = []
    train_loss = []
    for line in lines:
        if line[:5] == 'epoch':
            line = line.split()
            if line[-1] == 'training':
                train_loss.append([])
            if line[-2] == 'mean_val_loss=':
                val_loss.append(float(line[-1]))
        if line[:5] == 'iters':
            train_loss[-1].append(float(line.split('=')[-1]))
    train_loss = np.array(train_loss)
    val_loss = np.array(val_loss)
    return train_loss, val_loss


class test(object):
    def __init__(self, model, train_name, test_loader):
        self.data = 1
        self.testset_path = './dataset/data/test'
        self.test_loader = test_loader
        self.model = model
        self.output_folder = '../output/%s/tor_pred' % train_name
        self.coo_folder = '../output/%s/coo_pred' % train_name
        self.cb_folder = '../output/%s/cb_pred' % train_name
        self.model_dir = '../output/%s/models' % train_name
        self.train_name = train_name
        pathlib.Path(self.output_folder).mkdir(parents=True, exist_ok=True)
        pathlib.Path(self.coo_folder).mkdir(parents=True, exist_ok=True)
        pathlib.Path(self.cb_folder).mkdir(parents=True, exist_ok=True)

        self.rmsds = []
        self.gdts = []
        self.ramas = []

    def test_model(self, model_name):
        self.model.load_model(self.train_name, model_name)
        self.model.eval()
        self.model.training = False
        output_path = os.path.join(self.output_folder, model_name)
        coo_path = os.path.join(self.coo_folder, model_name)
        cb_path = os.path.join(self.cb_folder, model_name)
        pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
        pathlib.Path(coo_path).mkdir(parents=True, exist_ok=True)
        pathlib.Path(cb_path).mkdir(parents=True, exist_ok=True)

        with torch.no_grad():
            for inputs, _, filenames, lengths in self.test_loader:
                inputs = inputs[0].cuda(non_blocking=True)
                outputs = self.model(inputs, lengths).squeeze(
                    1).transpose(0, 1)
                outputs = outputs.data.cpu().numpy()

                last = 0
                for filename, l_ in zip(filenames, lengths):
                    filename = filename[0]
                    next_ = last + (l_-1)
                    out = outputs[:, last: next_]
                    np.save(os.path.join(output_path, filename), out)
                    last = next_

                    coo = np.load(os.path.join(
                        self.testset_path, 'coo', '%s.npy' % filename))
                    with open(os.path.join(self.testset_path, 'seq', '%s.txt' % filename)) as f:
                        seq = f.read()
                    cb = np.load(os.path.join(
                        self.testset_path, 'cb', '%s.npy' % filename))

                    ca = coo[1::4]
                    coo_, cb_ = bbrebuild(ca, out, seq)
                    np.save(os.path.join(coo_path, filename), coo_)
                    np.save(os.path.join(cb_path, filename), cb_)

                    rmsd, gdt, rama = cal_criteria(coo, cb, coo_, cb_, seq)
                    self.rmsds.append(rmsd)
                    self.gdts.append(gdt)
                    self.ramas.append(rama)

    def ensemble_output(self, model_names):
        ensemble_path = os.path.join(self.output_folder, 'ensemble')
        coo_path = os.path.join(self.coo_folder, 'ensemble')
        cb_path = os.path.join(self.cb_folder, 'ensemble')
        pathlib.Path(ensemble_path).mkdir(parents=True, exist_ok=True)
        pathlib.Path(coo_path).mkdir(parents=True, exist_ok=True)
        pathlib.Path(cb_path).mkdir(parents=True, exist_ok=True)

        output_paths = [os.path.join(
            self.output_folder, str(model_name)) for model_name in model_names]
        filenames = os.listdir(output_paths[0])
        for filename in filenames:
            output = [np.load(os.path.join(output_path, filename))
                      for output_path in output_paths]
            ensemble_output = np.mean(np.array(output), axis=0)
            np.save(os.path.join(ensemble_path, filename), ensemble_output)

            filename = filename[:-4]
            coo = np.load(os.path.join(self.testset_path,
                                       'coo', '%s.npy' % filename))
            with open(os.path.join(self.testset_path, 'seq', '%s.txt' % filename)) as f:
                seq = f.read()
            cb = np.load(os.path.join(
                self.testset_path, 'cb', '%s.npy' % filename))

            ca = coo[1::4]
            coo_, cb_ = bbrebuild(ca, ensemble_output, seq)
            np.save(os.path.join(coo_path, filename), coo_)
            np.save(os.path.join(cb_path, filename), cb_)

            rmsd, gdt, rama = cal_criteria(coo, cb, coo_, cb_, seq)
            self.rmsds.append(rmsd)
            self.gdts.append(gdt)
            self.ramas.append(rama)

    def test_top_models(self, top_num=3, ensemble=True):
        _, val_loss = loss_from_log(self.train_name)
        top_models_index = np.argsort(val_loss)[:top_num]

        for i in top_models_index:
            self.test_model(str(i))

        if ensemble:
            self.ensemble_output(top_models_index)

        self.rmsds = np.array(self.rmsds)
        self.gdts = np.array(self.gdts)
        self.ramas = np.array(self.ramas)

        np.save('../results/rmsd/%s.npy' % self.train_name, self.rmsds)
        np.save('../results/gdt/%s.npy' % self.train_name, self.gdts)
        np.save('../results/rama/%s.npy' % self.train_name, self.ramas)
