from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
from utils import one_hot

class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, previous_dir, scale=1, mask_suffix=' 7, '):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.previous_dir = previous_dir
        self.scale = scale
        self.mask_suffix = mask_suffix
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))
        img_nd = np.array(pil_img)
        img_nd = np.expand_dims(img_nd, axis=2)
        # # if len(img_nd.shape) >= 2:
        # #     img_nd = np.expand_dims(img_nd, axis=13)
        # # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        # img = img_trans.squeeze(0)
        # img = img_nd.squeeze(0)
        # if img_trans.shape != (2, 256, 256):#通道数
        #     img_trans = np.stack((img,)*2, axis=0)
        # if img_trans.max() > 1:
        #     img_trans = img_trans / 255
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    def mask_preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))

        # palette = [[0], [20], [39], [58], [77], [96], [115], [134], [153], [172], [191],
        #            [210], [229], [248]]
        palette = [[0], [255]]

        img_nd = np.array(pil_img)
        img_nd = np.expand_dims(img_nd, axis=2)
        img_nd = one_hot.mask_to_onehot(img_nd, palette)
        # if len(img_nd.shape) >= 2:
        #     img_nd = np.expand_dims(img_nd, axis=13)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        # if img_trans.max() > 1:
        #     img_trans = img_trans / 255

        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        # mask_file = glob(self.masks_dir + idx + self.mask_suffix + '.*')
        img_file = glob(self.imgs_dir + idx + '.*')
        mask_file = glob(self.masks_dir + idx + '.*')
        previous_file = glob(self.previous_dir + idx + '.*')

        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        assert len(previous_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {previous_file}'

        mask = Image.open(mask_file[0])
        img = Image.open(img_file[0])
        previous = Image.open(previous_file[0])

        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'


        mask = self.preprocess(mask, self.scale)
        img = self.preprocess(img, self.scale)
        previous = self.preprocess(previous, self.scale)

        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'mask': torch.from_numpy(mask).type(torch.FloatTensor),
            'previous': torch.from_numpy(previous).type(torch.FloatTensor)
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, imgs_dir, masks_dir, scale=1):
        super().__init__(imgs_dir, masks_dir, scale, mask_suffix='_mask')
