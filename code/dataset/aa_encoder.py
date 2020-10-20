import numpy as np


class AminoacidEncoder(object):
    def __init__(self, aa_format='property'):
        self.aa_list = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                        'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
        self.index = {}
        for aa in self.aa_list:
            self.index[aa] = self.aa_list.index(aa)

        if aa_format == 'onehot':
            self.encoder = np.eye(20)

        elif aa_format == 'property':
            self.hydropathicity = [1.8, 2.5, -3.5, -3.5, 2.8, -0.4, -3.2, 4.5, -3.9, 3.8,
                                   1.9, -3.5, -1.6, -3.5, -4.5, -0.8, -0.7, 4.2, -0.9, -1.3]
            self.bulkiness = [11.5, 13.46, 11.68, 13.57, 19.8, 3.4, 13.69, 21.4, 15.71, 21.4,
                              16.25, 12.82, 17.43, 14.45, 14.28, 9.47, 15.77, 21.57, 21.67, 18.03]
            self.flexibility = [14.0, 0.05, 12.0, 5.4, 7.5, 23.0, 4.0, 1.6, 1.9, 5.1,
                                0.05, 14.0, 0.05, 4.8, 2.6, 19.0, 9.3, 2.6, 0.05, 0.05]
            self.property_norm()

            self.encoder = np.stack([self.hydropathicity,
                                     self.bulkiness,
                                     self.flexibility]).T.astype('float32')

    def property_norm(self):
        self.hydropathicity = (5.5 - np.array(self.hydropathicity)) / 10
        self.bulkiness = np.array(self.bulkiness) / max(self.bulkiness)
        self.flexibility = (25 - np.array(self.flexibility)) / 25

    def encode(self, seq):
        indexs = np.array([self.index[aa] for aa in seq])
        return self.encoder[indexs]
