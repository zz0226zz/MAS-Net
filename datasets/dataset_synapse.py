import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample


class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name+'.npz')
            data = np.load(data_path)
            image, label = data['image'], data['label']
        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
            data = h5py.File(filepath)
            image, label = data['image'][:], data['label'][:]

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample

#
# import os
# import numpy as np
# import h5py
# import random
# from torch.utils.data import Dataset
#
#
# class Synapse_dataset(Dataset):
#     def __init__(self, base_dir, list_dir, split, n_way=2, n_shot=5, n_query=15, transform=None):
#         self.transform = transform
#         self.split = split
#         self.n_way = n_way  # 多少类别
#         self.n_shot = n_shot  # 每个类别的支持样本数
#         self.n_query = n_query  # 每个类别的查询样本数
#
#         # 读取样本列表
#         self.sample_list = open(os.path.join(list_dir, self.split + '.txt')).readlines()
#         self.sample_list = [line.strip() for line in self.sample_list]
#         self.data_dir = base_dir
#
#         # 进行 few-shot 数据划分
#         if self.split == "train":
#             self.support_set, self.query_set = self._split_few_shot_data()
#
#     def _split_few_shot_data(self):
#         """划分 few-shot 训练的 support 和 query 数据"""
#         random.shuffle(self.sample_list)  # 先打乱数据
#         class_dict = {}  # 存储不同类别的样本
#
#         # 统计类别
#         for sample in self.sample_list:
#             class_label = int(sample.split('_')[-1])  # 假设类别信息在文件名中
#             if class_label not in class_dict:
#                 class_dict[class_label] = []
#             class_dict[class_label].append(sample)
#
#         support_set, query_set = [], []
#
#         # 选择 n_way 类别，每个类别选择 n_shot 作为 support，其余作为 query
#         selected_classes = random.sample(list(class_dict.keys()), self.n_way)
#         for class_label in selected_classes:
#             samples = class_dict[class_label]
#             random.shuffle(samples)  # 再次打乱
#             support_set.extend(samples[:self.n_shot])
#             query_set.extend(samples[self.n_shot:self.n_shot + self.n_query])  # 选择查询样本
#
#         return support_set, query_set
#
#     def __len__(self):
#         return len(self.query_set)  # 只返回查询集大小
#
#     def __getitem__(self, idx):
#         """返回 support 和 query 样本"""
#         if self.split == "train":
#             # 交替返回 support 和 query
#             if idx < len(self.support_set):
#                 slice_name = self.support_set[idx]
#                 support_or_query = "support"
#             else:
#                 slice_name = self.query_set[idx - len(self.support_set)]
#                 support_or_query = "query"
#
#             data_path = os.path.join(self.data_dir, slice_name + '.npz')
#             data = np.load(data_path)
#             image, label = data['image'], data['label']
#
#         else:  # 测试数据
#             vol_name = self.sample_list[idx]
#             filepath = os.path.join(self.data_dir, f"{vol_name}.npy.h5")
#             data = h5py.File(filepath, 'r')
#             image, label = data['image'][:], data['label'][:]
#             support_or_query = "test"
#
#         sample = {'image': image, 'label': label, 'case_name': slice_name, 'type': support_or_query}
#
#         if self.transform:
#             sample = self.transform(sample)
#
#         return sample
