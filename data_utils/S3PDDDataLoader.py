import os
import numpy as np
import warnings
import _pickle as pickle

from torch.utils.data import Dataset

warnings.filterwarnings('ignore')

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

class S3PDDDataLoader(Dataset):
    def __init__(self, path, args, split='train'):        
        self.path = path
        self.npoints = args.num_point
        self.split = split
        self.classes = {"HLT": 0, "BLD": 1, "PVY": 2}
        self.num_category = len(self.classes)
        
        self.data_path = os.path.join(self.path, self.split + '.pkl')
        
        with open(self.data_path, "rb") as fp:
            self.id_list = pickle.load(fp)
            self.box2d_list = pickle.load(fp)
            self.input_list = pickle.load(fp)
            self.type_list = pickle.load(fp)
            self.conf_list = pickle.load(fp)
            self.prob_list = pickle.load(fp)

        if np.array(self.prob_list).shape[1] != self.num_category:
            raise ValueError(f"Number of classes in prob_list ({np.array(self.prob_list).shape[1]}) does not match num_category ({self.num_category})")

    def __len__(self):
        return len(self.id_list)
    
    def __getitem__(self, index):
        point_set = self.input_list[index][:, 0:3]        
        point_set = farthest_point_sample(point_set, self.npoints)
        point_set = pc_normalize(point_set)
        prob_set = self.prob_list[index]
        if self.split != 'test':
            label = self.classes[self.type_list[index]]
            # label = np.zeros(self.num_category, dtype=np.float32)
            # label[self.classes[self.type_list[index]]] = 1.0
            return point_set, prob_set, label
        else:
            return point_set, prob_set


if __name__ == '__main__':
    import torch

    data = S3PDDDataLoader('/mnt/guanabana/raid/home/jia021/S3-PDD/runs_test/val/exp/velodyne_pkl', args=[], split='val')
    DataLoader = torch.utils.data.DataLoader(data, batch_size=12, shuffle=True)
    for point, label in DataLoader:
        print(point.shape)
        print(label.shape)