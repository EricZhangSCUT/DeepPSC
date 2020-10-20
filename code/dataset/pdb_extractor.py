# coding=utf-8
import logging
import numpy as np
import os
import pathlib


class PDBFileExtractor(object):
    def __init__(self, atoms_list=['N', 'CA', 'C', 'O'], logger=None):
        self.alphabet = {'ALA': 'A', 'PHE': 'F', 'CYS': 'C', 'ASP': 'D', 'ASN': 'N',
                         'GLU': 'E', 'GLN': 'Q', 'GLY': 'G', 'HIS': 'H', 'LEU': 'L',
                         'ILE': 'I', 'LYS': 'K', 'MET': 'M', 'PRO': 'P', 'ARG': 'R',
                         'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'}
        self.logger = logger
        self.atoms_list = atoms_list

    def extract_lines(self, pdb_file):
        with open(pdb_file) as f:
            lines = f.readlines()
        chains = {}
        for line in lines:
            line_split = line.split()
            if line_split[0] == "ATOM" and line_split[2] in self.atoms_list:
                if chains.get(line[21]) is None:
                    chains[line[21]] = []
                chains[line[21]].append(line)
        return chains

    def check_completeness(self, lines):
        atoms_list_len = len(self.atoms_list)
        if len(lines) % atoms_list_len:
            return False
        for i in range(atoms_list_len):
            for line in lines[i::atoms_list_len]:
                if line.split()[2] != self.atoms_list[i]:
                    return False
        return True

    def remove_repeat(self, lines):
        '''remove repeat atoms by occupancy'''
        remove_list = []
        i = 1
        while i < len(lines):
            line_i = lines[i].split()
            if lines[i - 1].split()[2] == line_i[2] and lines[i - 1].split()[5] == line_i[5]:
                repeat_atoms = []
                repeat_atoms.append(i-1)
                repeat_atoms.append(i)
                temp_idx = i + 1

                while temp_idx < len(lines) and\
                        lines[temp_idx].split()[2] == line_i[2] and\
                        lines[temp_idx].split()[5] == line_i[5]:
                    repeat_atoms.append(temp_idx)
                    temp_idx += 1

                i = temp_idx

                max_idx = repeat_atoms[0]
                max_ = float(lines[max_idx][56:60])
                for idx in repeat_atoms:
                    if float(lines[idx][56:60]) >= max_:
                        max_ = float(lines[idx][56:60])
                        max_idx = idx
                repeat_atoms.remove(max_idx)
                for idx in repeat_atoms:
                    remove_list.append(idx)
            else:
                i += 1

        for i in remove_list[::-1]:
            del lines[i]
        return lines

    def get_coo_and_seq(self, chain):
        coo = []
        seq = ''
        for line in chain:
            coo.append([line[27:38].strip(),
                        line[38:46].strip(),
                        line[46:54].strip()])
            seq += self.alphabet[line[17:20]]
        coo = np.array(coo).astype('float32')
        seq = seq[::len(self.atoms_list)]
        return coo, seq

    def extract(self, pdb_file):
        chains = self.extract_lines(pdb_file)
        coo = {}
        seq = {}
        for chain_id in chains.keys():
            chain = self.remove_repeat(chains[chain_id])
            if self.check_completeness(chain):
                coo[chain_id], seq[chain_id] = self.get_coo_and_seq(chain)
            else:
                filename = pdb_file.split('/')[-1][:-4]
                if self.logger:
                    self.logger.info("Missing atoms in chain %s of %s!!!" %
                                     (chain_id, filename))
                else:
                    print("Missing atoms in chain %s of %s!!!" %
                          (chain_id, filename))
        return coo, seq


if __name__ == "__main__":
    dataset = 'nr40'
    pdb_path = './data/%s/pdb' % dataset

    coo_path = './data/%s/coo' % dataset
    seq_path = './data/%s/seq' % dataset
    pathlib.Path(coo_path).mkdir(parents=True, exist_ok=True)
    pathlib.Path(seq_path).mkdir(parents=True, exist_ok=True)

    log_path = './PDB_extract_logs'
    pathlib.Path(log_path).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, filename='./PDB_extract_logs/%s.log' % dataset,
                        filemode='w', format='%(message)s')
    logger = logging.getLogger(__name__)

    extractor = PDBFileExtractor(logger)

    for filename in os.listdir(pdb_path):
        coo, seq = extractor.extract(os.path.join(pdb_path, filename))
        for chain_id in coo.keys():
            np.save(os.path.join(coo_path, "%s_%s.npy" %
                                 (filename[:-4], chain_id)), coo)
            with open(os.path.join(seq_path, "%s_%s.txt" % (filename[:-4], chain_id)), "w") as writer:
                writer.write(seq)
