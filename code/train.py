import torch
import torch.nn as nn
from model import DeepPSC
import dataset
from torch.utils.data import DataLoader
import pathlib
import os
import time
import logging
import argparse
from test import test
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--loc_dim', type=int, default=512)
parser.add_argument('--glo_dim', type=int, default=1024)
parser.add_argument('--tgt_dim', type=int, default=4)
parser.add_argument('--set', type=str, default=None)
parser.add_argument('--note', type=str, default=None)
parser.add_argument('--epoch', type=int, default=30)
args = parser.parse_args()


train_name = '-'.join([arg for arg in [args.set, args.note] if arg])

save_dir = '../output/%s' % train_name
model_dir = '../output/%s/model' % train_name
dirs = [save_dir, model_dir]
for folder in dirs:
    pathlib.Path(folder).mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, filename='../logs/log_%s.txt' %
                    train_name, filemode='w', format='%(message)s')
logger = logging.getLogger(__name__)

dataset = dataset.Image2Tor
inp_dim = 5

train_dataset = dataset(
    dataset='nr40', file_list='train_%s' % args.set if args.set else 'train')
val_dataset = dataset(
    dataset='nr40', file_list='val_%s' % args.set if args.set else 'val')
test_dataset = dataset('test', with_target=False)

train_loader = DataLoader(dataset=train_dataset, shuffle=True,
                          num_workers=32, pin_memory=True)
val_loader = DataLoader(dataset=val_dataset, num_workers=32, pin_memory=True)
test_loader = DataLoader(dataset=test_dataset, num_workers=32, pin_memory=True)

model = DeepPSC(dims=[inp_dim, args.loc_dim, args.glo_dim, args.tgt_dim])

logger.info('-----Model-----')
logger.info(model)
logger.info('-----Model-----\n\n')

loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), 3e-4, weight_decay=10e-6)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=len(train_dataset))
warmup_epochs = 3


if __name__ == '__main__':
    total_iters = 0
    for epoch in range(args.epoch):
        epoch_start_time = time.time()
        epoch_iter = 0
        losses = 0
        logger.info('epoch %s training' % epoch)

        for inp, tgt, filename, lengths in train_loader:
            optimizer.zero_grad()
            inp = inp[0].cuda(non_blocking=True)
            tgt = tgt[0].cuda(non_blocking=True)

            out = model(inp, lengths).squeeze(1).transpose(0, 1)
            loss = loss_function(out, tgt)
            losses += loss.item()

            total_iters += 1
            epoch_iter += 1

            loss.backward()
            optimizer.step()

            if total_iters % 20 == 0:
                logger.info('iters %s train_loss=%s' %
                            (total_iters, losses / 20))
                losses = 0

            if total_iters % 1000 == 0:
                model.save_model(os.path.join(model_dir, 'last_model'))

            if epoch >= warmup_epochs:
                scheduler.step()

        logger.info('epoch %s validating...' % epoch)
        model.save_model(os.path.join(model_dir, '%s_model' % epoch))
        model.eval()

        with torch.no_grad():
            losses = 0

            for inp, tgt, filename, lengths in val_loader:
                inp = inp[0].cuda(non_blocking=True)
                tgt = tgt[0].cuda(non_blocking=True)

                out = model(inp, lengths).squeeze(1).transpose(0, 1)

                loss = loss_function(out, tgt)
                losses += loss.item()

            logger.info('epoch %d, mean_val_loss= %f' %
                        (epoch, losses / len(val_dataset)))

        model.train()

        logger.info('End of epoch %d  \t Time Taken: %d sec' %
                    (epoch, time.time() - epoch_start_time))

    tester = test(model, train_name=train_name, test_loader=test_loader)
    tester.test_model('29')

    rmsd = np.round(np.mean(np.array(tester.rmsds), axis=0), 3)
    gdt = np.round(
        np.mean(np.mean(np.array(tester.gdts), axis=-1), axis=0)*100, 3)
    rama = np.round(np.mean(np.array(tester.ramas), axis=0)*100, 3)

    logger.info('\n\n--Results--')
    logger.info('-RMSD-')
    logger.info(rmsd)
    logger.info('-GDT-')
    logger.info(gdt)
    logger.info('-RAMA-')
    logger.info(rama)
