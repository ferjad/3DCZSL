import sys
import os
import os.path as osp
from os.path import join as ospj
from collections import OrderedDict, defaultdict
import json

import h5py
import numpy as np
import torch

from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

sys.path.insert(0, osp.dirname(__file__) + '/..')
from shaper.models.pointnet2 import functions as _F
from shaper.utils.pc_util import normalize_points as normalize_points_np


class PartNetInsSegGlobal(Dataset):
    base_classes = ["Bag", "Bottle", "Keyboard", "Knife", "Microwave", "StorageFurniture", "Vase", "Earphone", "Faucet",
                "Display", "Hat", "Bed", "Chair", "Clock", "Lamp", "Table"]
    
    validation_classes = ['Bowl', 'Dishwasher']
    fully_composable_classes = ["Bowl", "Dishwasher", "Door", "Laptop", "Mug", "Refrigerator", "Scissors", "TrashCan"]

    partial_classes = []
    all_shapes = sorted(list(set(base_classes+fully_composable_classes+partial_classes)))
    def __init__(self,
                 root_dir,
                 split,
                 train_objects,
                 eval_objects,
                 normalize=True,
                 transform=None,
                 zero_shot = False,
                 shape='',
                 stage1='',
                 level=-1,
                 cache_mode=False,
                 downsample = None,
                 seen_only = False,
                 pose = False
                 ):
        self.root_dir = root_dir
        self.split = split #train, val or test
        self.normalize = normalize
        self.transform = transform
        self.zero_shot = zero_shot
        self.cat_file = ospj(self.root_dir,'shape_names.txt')
        self.shape_levels = self._load_cat_file()
        self.cache_mode = cache_mode
        self.downsample = downsample
        self.seen_only = seen_only
        self.train_objects = train_objects
        self.eval_objects = eval_objects
        
        # Mask of seen objects, used in cross entropy
        if self.seen_only:
            self.seen_lookup = {}
            idx_replace = 0
            for idx, shape in enumerate(self.all_shapes):
                if shape not in self.fully_composable_classes:
                    self.seen_lookup[idx] = idx_replace
                    idx_replace+=1
                    print(shape)
        
        if self.downsample:
            self.downsample_idx = np.arange(downsample)
        self.folder_list = self._prepare_file_list()
        self.class_map = ospj(root_dir, 'shape_to_parts')
        self.class2mapping, self.class2parts, self.part2idx = self.get_parts_info(self.all_shapes) # TODO: Choose shapes from yaml
        self.class_names = list(self.class2mapping.keys())
        self.class_to_ind_map = {c: i for i, c in enumerate(self.class_names)}
        self.idx2part = dict((v,k) for k,v in self.part2idx.items())
        self.cache = defaultdict(list)
        self.meta_data = []
        self._load_data()
        self.pose = pose
        
        # For shape level inference
        shape_masks = []
        for key, cls_mapping in self.class2mapping.items():
            local_label_map = np.zeros(len(self.part2idx), dtype=np.uint8)
            cls_mapping = cls_mapping[1:]
            local_label_map[cls_mapping] = 1
            shape_masks.append(local_label_map)

        self.shape_masks = np.array(shape_masks, dtype=np.uint8)
        
        length = len(self.meta_data)
        print(f'{self.__class__.__name__} with {length} shapes')
        if self.cache_mode:
            length = len(self.cache['points'])
            print(f'{self.__class__.__name__} with {length} cached elements')

    def file_to_parts(self, filename):
        class_names = []
        with open(filename) as f:
            for line in f.readlines():
                line = line.strip().split(' ')[1].split('/')[-1]
                class_names.append(line)
        return class_names

    def get_parts_info(self, shape_list):
        '''
            Gets the global part information for last levels of heirarchy
            returns
                class2mapping: A dictionary of array, mapping original labels to global labels
                class2parts: A dictionary of list of parts for every shape
                part2idx: A dictionary of parts to corresponding id in global label
        '''
        class2parts = {}
        part2idx = {}
        class2mapping = {}
        part2idx['other'] = 0
        count = 1
        for shape in shape_list:
            class_mapping = (ospj(self.class_map, shape+'.txt'))
            parts = self.file_to_parts(class_mapping)
            class2parts[shape] = parts
            part_mapping = [0]
            for part in parts:
                if part not in part2idx:
                    lookup = count
                    part2idx[part] = lookup
                    count+=1
                else:
                    lookup = part2idx[part]
                part_mapping.append(lookup)
            part_mapping = np.array(part_mapping)
            class2mapping[shape] = part_mapping

        return class2mapping, class2parts, part2idx

    def _load_cat_file(self):
        # Assume that the category file is put under root_dir.
        with open(self.cat_file, 'r') as fid:
            shape_levels = OrderedDict()
            for line in fid:
                shape, levels = line.strip().split('\t')
                levels = tuple([int(l) for l in levels.split(',')])[-1] #split by , to count levels
                shape_levels[shape] = levels
            return shape_levels

    def _prepare_file_list(self):
        '''
        returns folders to load based on shape and level,
        '''
        folder_list = []
        for shape, l in self.shape_levels.items():
            # Check for folder in allowed list
            # When split is train
            if self.split == 'train' and shape not in self.train_objects:
                print(f'Skipping {shape} for {self.split} split')
                continue
            elif self.split != 'train' and shape not in self.train_objects and shape not in self.eval_objects:
                print(f'Skipping {shape} for {self.split} split')
                continue
            folder_list.append(f'{shape}-{l}')
        return sorted(folder_list)

    def _load_data(self):
        def load_from_file(folder_path, files, start = 'train', end = 'h5'):
            for fname in files:
                if fname.startswith(start) and fname.endswith(end):

                    data_path = ospj(folder_path, fname)
                    print(f'Loading {data_path}')

                    with h5py.File(data_path, mode = 'r') as f:
                        num_samples = f['pts'].shape[0]
                        print("folder path: ", folder_path, " ", num_samples)
                        for current in range(num_samples):
                            self.meta_data.append((data_path, current))
                        
                        if self.cache_mode:
                            # point cloud [N, 10000, 3]
                            pts = f['pts'][:]
                            gt_label = f['gt_label'][:]

                            if self.downsample:
                                pts = pts[:, self.downsample_idx, :]
                                gt_label = gt_label[:, self.downsample_idx]
                            self.cache['points'].append(pts)
                            # semantics class [N, 10000]
                            self.cache['gt_label'].append(gt_label)
                            #  semantics global class [N, 10000]
                            global_label = np.take(class_mapping,gt_label)
                            self.cache['gt_label_global'].append(global_label)
                            # instance idx [N, 200, 10000]
                            self.cache['gt_mask'].append(f['gt_mask'][:])
                            # valid class indicator [N, 200]
                            self.cache['gt_valid'].append(f['gt_valid'][:])
                            # valid class indicator [N, 10000]
                            self.cache['gt_other_mask'].append(f['gt_other_mask'][:])
                            # shape class label
                            self.cache['gt_class_label'].append(cls_label)

        data_folders = ['ins_seg_h5_for_detection_main',
                       'unseen_merged', 'ins_seg_h5_for_detection_unseen']
        for folder in self.folder_list:
            print(f'Getting folder {folder}')
            # Get folder
            class_name = folder.split('-')[0]
            class_mapping = self.class2mapping[class_name]
            cls_label = self.class_to_ind_map[class_name]
            # If not zero-shot, only focus on one folder
            if not self.zero_shot:
                folder_path = ospj(self.root_dir, data_folders[0], folder)
                files  = os.listdir(folder_path)
                load_from_file(folder_path, files, start = self.split, end = 'h5')
            # If zero_shot
            else:
                #  for base classes, everything is in the first folder
                if class_name in self.base_classes:
                    print('In Base class')
                    folder_path = ospj(self.root_dir, data_folders[0], folder)
                    files  = os.listdir(folder_path)
                    load_from_file(folder_path, files, start = self.split, end = 'h5')
                # for composable classes, only unseen data available in second folder
                if class_name in self.fully_composable_classes:
                    print('In Composable class')
                    if class_name in self.validation_classes:
                        folder_path = ospj(self.root_dir, data_folders[2], folder)
                        files  = os.listdir(folder_path)
                        load_from_file(folder_path, files, start = self.split, end = 'h5')
                    else:
                        folder_path = ospj(self.root_dir, data_folders[1], folder)
                        files  = os.listdir(folder_path)
                        load_from_file(folder_path, files, start = 'unseen', end = 'h5')

        #list to nparray
        for k,v in self.cache.items():
            if k == 'gt_class_label':
                self.cache[k] = np.stack(v)
            else:
                self.cache[k] = np.concatenate(v, axis = 0)

    def get_item_cache(self, index):
        points = self.cache['points'][index]
        # (10000, )
        gt_label = self.cache['gt_label'][index]
        gt_label_global = self.cache['gt_label_global'][index]
        # (200, 10000)
        gt_mask = self.cache['gt_mask'][index]
        # (200,)
        gt_valid = self.cache['gt_valid'][index]
        # (10000, )
        gt_other_mask = self.cache['gt_other_mask'][index]

        # 0 for ignore
        # Combine int one num_ins*10000 array for cross entropy
        gt_all_mask = np.concatenate([gt_other_mask[None, :], gt_mask], axis=0)
        ins_id = gt_all_mask.argmax(axis=0)

        #  semantics global class [N, 10000]
        path, idx = self.meta_data[index]
        class_name = path.split('/')[-2].split('-')[0]
        class_mapping = self.class2mapping[class_name]
        gt_label_global = np.take(class_mapping, gt_label)

        cls_label = self.class_to_ind_map[class_name]
        if self.seen_only:
            cls_label = self.seen_lookup[cls_label]
            
        sample_id = self.get_sampleid(cls_label, idx)

        local_label_map = np.zeros(len(self.part2idx), dtype=np.uint8)
        cls_mapping = class_mapping[1:]
        local_label_map[cls_mapping] = 1

        if self.normalize:
            points = normalize_points_np(points)
        if self.transform is not None:
            points, ins_id = self.transform(points, ins_id)

        out_dict = dict(
            points=np.transpose(points, (1, 0)),
            ins_id=ins_id,
            gt_mask=np.array(gt_mask, dtype=np.uint8),
            gt_valid=np.array(gt_valid, dtype=np.uint8),
            gt_label=np.array(gt_label, dtype=np.uint8),
            gt_label_global=np.array(gt_label_global, dtype=np.uint8),
            gt_other_mask=np.array(gt_other_mask, dtype=np.uint8),
            gt_class_label=cls_label,
            local_label_map=local_label_map,
            sample_id=sample_id,
            shape_masks=self.shape_masks
        )

        return out_dict
    
    def get_item_file(self, index):
        path, idx = self.meta_data[index]
        
        with h5py.File(path, mode = 'r') as f:
            num_samples = f['pts'].shape[0]
            assert idx <= num_samples
            # point cloud [N, 10000, 3]
            points = f['pts'][idx]
            # semantics class [N, 10000]
            gt_label = f['gt_label'][idx]
            # instance idx [N, 200, 10000]
            gt_mask = f['gt_mask'][idx]
            # valid class indicator [N, 200]
            gt_valid = f['gt_valid'][idx]
            # valid class indicator [N, 10000]
            gt_other_mask = f['gt_other_mask'][idx]
        
        #  semantics global class [N, 10000]
        class_name = path.split('/')[-2].split('-')[0]
        class_mapping = self.class2mapping[class_name]
        gt_label_global = np.take(class_mapping, gt_label)

        cls_label = self.class_to_ind_map[class_name]
        if self.seen_only:
            cls_label = self.seen_lookup[cls_label]
        sample_id = self.get_sampleid(cls_label, index)

        local_label_map = np.zeros(len(self.part2idx), dtype=np.uint8)
        cls_mapping = class_mapping[1:]
        local_label_map[cls_mapping] = 1
        # 0 for ignore
        # Combine int one num_ins*10000 array for cross entropy
        gt_all_mask = np.concatenate([gt_other_mask[None, :], gt_mask], axis=0)
        ins_id = gt_all_mask.argmax(axis=0)

        if self.normalize:
            points = normalize_points_np(points)
        if self.transform is not None:
            points, ins_id = self.transform(points, ins_id)

        if self.downsample:
            points = points[self.downsample_idx, :]
            gt_label = gt_label[self.downsample_idx]
            gt_label_global = gt_label_global[self.downsample_idx]

        # if self.pose:
        #    _, lrf = estimate_pointcloud_local_coord_frames(torch.tensor(points).unsqueeze(0))
        #    lrf = torch.flatten(lrf, start_dim=2) # flatten 3x3 to 9dim vec

        out_dict = dict(
            points=np.transpose(points, (1, 0)),
            ins_id=ins_id,
            gt_mask=np.array(gt_mask,dtype=np.uint8),
            gt_valid=np.array(gt_valid,dtype=np.uint8),
            gt_label=np.array(gt_label,dtype=np.uint8),
            gt_label_global=np.array(gt_label_global,dtype=np.uint8),
            gt_other_mask=np.array(gt_other_mask,dtype=np.uint8),
            gt_class_label=cls_label,
            local_label_map=local_label_map,
            sample_id=sample_id,
            shape_masks=self.shape_masks
        )

        return out_dict
    
    def __getitem__(self, index):
        if self.cache_mode:
            return self.get_item_cache(index)
        else:
            return self.get_item_file(index)
    
    def __len__(self):
        return len(self.meta_data)

    def sampleid_tostring(self, sample_id):
        idx = sample_id % 100000
        cls_label = (sample_id - idx)//100000
        return cls_label, idx

    def get_sampleid(self, cls_label, idx):
        sample_id = cls_label * 100000 + idx
        return sample_id
        
def collate(batch, num_centroids, radius, num_neighbours,
            with_renorm, with_resample, with_shift, sample_method):
    data_batch = default_collate(batch)
    with torch.no_grad():
        xyz = data_batch.get('points').cuda(non_blocking=True)
        # ins_id, (batch_size, length)
        ins_id = data_batch.get('ins_id').cuda(non_blocking=True)
        batch_size, length = ins_id.size()

        # sample new points
        # (batch_size, num_centroids)
        assert sample_method in ['RND', 'FPS', 'LS', 'WBS']
        if sample_method == 'RND':
            centroid_index = torch.randint(low=0, high=length, size=(batch_size, num_centroids), device=ins_id.device)
        elif sample_method == 'LS':
            linspace = torch.linspace(-1, 1, steps=int(1/radius), device=ins_id.device)
            pseudo_centroids = torch.stack(torch.meshgrid(linspace, linspace, linspace), dim=0).view(1, 3, -1)
            # pdist, [batch_size, num_centroids, length]
            pdist = (xyz.unsqueeze(2) - pseudo_centroids.unsqueeze(3)).norm(dim=1)
            # (batch_size, num_centroids)
            _, centroid_index = pdist.min(dim=2)
        elif sample_method == 'WBS':
            num_centroids *= 2
            centroid_index = torch.randint(low=0, high=length, size=(batch_size, num_centroids), device=ins_id.device)
            # (batch_size, 3, num_centroids)
            centroid_xyz = _F.gather_points(xyz, centroid_index)
            # (batch_size, num_centroids, num_neighbours)
            neighbour_index, _ = _F.ball_query(xyz, centroid_xyz, radius, num_neighbours)
            # (batch_size, 3, num_centroids, num_neighbours)
            neighbour_xyz = _F.group_points(xyz, neighbour_index)

            neighbour_centroid_index = (neighbour_xyz - centroid_xyz.unsqueeze(-1)).abs().sum(1).argmin(dim=-1)

            resample_centroid_index = neighbour_index.gather(2, neighbour_centroid_index.unsqueeze(-1)).squeeze(-1)

            # neighbour_label, (batch_size, num_centroids, num_neighbours)
            neighbour_label = ins_id.gather(1, neighbour_index.view(batch_size, num_centroids*num_neighbours))
            neighbour_label = neighbour_label.view(batch_size, num_centroids, num_neighbours)

            # centroid_label, (batch_size, num_centroids, 1)
            centroid_label = ins_id.gather(1, resample_centroid_index).view(batch_size, num_centroids, 1)
            # (batch_size, num_centroids, num_neighbours)
            neighbour_centroid_dist = (neighbour_xyz - centroid_xyz.unsqueeze(-1)).norm(1)
            neighbour_centroid_dist = neighbour_centroid_dist * (neighbour_label != centroid_label.expand_as(neighbour_label)).float() + \
                                      neighbour_centroid_dist.max() * (neighbour_label == centroid_label.expand_as(neighbour_label)).float()
            # (batch_size, num_centroids)
            neighbour_centroid_dist, _ = neighbour_centroid_dist.min(dim=-1)
            _, select_centroid_index = neighbour_centroid_dist.topk(num_centroids//2, largest=False)
            centroid_index = centroid_index.gather(1, select_centroid_index)
        else:
            centroid_index = _F.farthest_point_sample(xyz, num_centroids)
        # (batch_size, 3, num_centroids)
        centroid_xyz = _F.gather_points(xyz, centroid_index)

        # (batch_size, num_centroids, num_neighbours)
        neighbour_index, _ = _F.ball_query(xyz, centroid_xyz, radius, num_neighbours)

        neighbour_index_purity, _ = _F.ball_query(xyz, centroid_xyz, 0.03, 64)
        neighbour_xyz_purity = _F.group_points(xyz, neighbour_index_purity)
        batch_size, num_centroids, num_neighbours = neighbour_index_purity.size()
        neighbour_label_purity = ins_id.gather(1, neighbour_index_purity.view(batch_size, num_centroids*num_neighbours))
        neighbour_label_purity = neighbour_label_purity.view(batch_size, num_centroids, num_neighbours)

        # (batch_size, 3, num_centroids, num_neighbours)
        neighbour_xyz = _F.group_points(xyz, neighbour_index)

        # TODO resample centroid_xyz and centroid_index var to stand for new centroid point
        # (batch_size, num_centroids)
        if with_resample:
            neighbour_centroid_index = torch.randint_like(centroid_index, low=0, high=num_neighbours)
        else:
            neighbour_centroid_index = (neighbour_xyz - centroid_xyz.unsqueeze(-1)).abs().sum(1).argmin(dim=-1)

        resample_centroid_index = neighbour_index.gather(2, neighbour_centroid_index.unsqueeze(-1)).squeeze(-1)

        # (batch_size, 3, num_centroids)
        resample_centroid_xyz = _F.gather_points(xyz, resample_centroid_index)

        # translation normalization
        centroid_mean = torch.mean(neighbour_xyz, -1).clone()
        neighbour_xyz -= centroid_mean.unsqueeze(-1)
        neighbour_xyz_purity -= centroid_mean.unsqueeze(-1)
        #neighbour_xyz -= centroid_xyz.unsqueeze(-1)
        resample_centroid_xyz -= centroid_xyz

        if with_renorm:
            norm_factor = neighbour_xyz.norm(dim=1).max()
            neighbour_xyz /= norm_factor
            resample_centroid_xyz /= norm_factor

        if with_shift:
            shift = neighbour_xyz.new(1).normal_(mean=0.0, std=0.01)
            neighbour_xyz += shift
            resample_centroid_xyz += shift

        batch_size, num_centroids, num_neighbours = neighbour_index.size()
        # neighbour_label, (batch_size, num_centroids, num_neighbours)
        neighbour_label = ins_id.gather(1, neighbour_index.view(batch_size, num_centroids*num_neighbours))
        neighbour_label = neighbour_label.view(batch_size, num_centroids, num_neighbours)

        # centroid_label, (batch_size, num_centroids, 1)
        centroid_label = ins_id.gather(1, resample_centroid_index).view(batch_size, num_centroids, 1)
        # ins_label, (batch_size, num_centroids, num_neighbours)
        ins_label = (neighbour_label == centroid_label.expand_as(neighbour_label)).long()
        valid_mask = ins_label.new_ones(ins_label.size())
        valid_mask[neighbour_label == 0] = 0
        valid_mask[centroid_label.expand_as(neighbour_label) == 0] = 0
        ins_label_purity = (neighbour_label_purity == centroid_label.expand_as(neighbour_label_purity)).long()
        purity_mask = (torch.sum(ins_label_purity,-1).float()/64 > 0.95)
        valid_mask[purity_mask.unsqueeze(-1).expand_as(neighbour_label) == 0] = 0
        valid_center_mask = purity_mask.new_ones(purity_mask.size())
        valid_center_mask[purity_mask == 0] = 0
        centroid_valid_mask = purity_mask.new_ones(purity_mask.size())
        centroid_valid_mask[purity_mask==0] = 0
        centroid_valid_mask[centroid_label.squeeze(-1) == 0] =0

        data_batch['neighbour_xyz'] = neighbour_xyz
        data_batch['neighbour_xyz_purity'] = neighbour_xyz_purity
        data_batch['neighbour_index'] = neighbour_index
        data_batch['centroid_xyz'] = resample_centroid_xyz
        data_batch['centroid_index'] = resample_centroid_index
        data_batch['centroid_label'] = centroid_label
        data_batch['centroid_valid_mask'] = centroid_valid_mask
        data_batch['neighbour_centroid_index'] = neighbour_centroid_index
        data_batch['ins_label'] = ins_label
        data_batch['valid_mask'] = valid_mask
        data_batch['valid_center_mask'] = valid_center_mask

        return data_batch
