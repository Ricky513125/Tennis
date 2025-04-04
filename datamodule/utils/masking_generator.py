import numpy as np


class TubeMaskingGenerator:
    def __init__(self, input_size, mask_ratio):
        self.frames, self.height, self.width = input_size
        self.num_patches_per_frame = self.height * self.width
        self.total_patches = self.frames * self.num_patches_per_frame
        self.num_masks_per_frame = int(mask_ratio * self.num_patches_per_frame)
        self.total_masks = self.frames * self.num_masks_per_frame

    def __repr__(self):
        repr_str = "Maks: total patches {}, mask patches {}".format(
            self.total_patches, self.total_masks
        )
        return repr_str

    def __call__(self):
        mask_per_frame = np.hstack(
            [
                np.zeros(self.num_patches_per_frame - self.num_masks_per_frame),
                np.ones(self.num_masks_per_frame),
            ]
        )
        # print('---mask_per_frame---') # (196,)
        # print(mask_per_frame.shape)
        np.random.shuffle(mask_per_frame)
        # print('---shape')
        # print('---mask_per_frame---', mask_per_frame.shape) # (196,)
        mask = np.tile(mask_per_frame, (self.frames, 1)).flatten()
        # print('---mask---', mask.shape) # (1568,)
        # print('TubeMasking!!!!')
        return mask


class RandomMaskingGenerator:
    def __init__(self, input_size, mask_ratio):
        self.frames, self.height, self.width = input_size
        self.num_patches_per_frame = self.height * self.width
        self.total_patches = self.frames * self.num_patches_per_frame
        self.num_masks_per_frame = int(mask_ratio * self.num_patches_per_frame)
        self.total_masks = self.frames * self.num_masks_per_frame

    def __repr__(self):
        repr_str = "Maks: total patches {}, mask patches {}".format(
            self.total_patches, self.total_masks
        )
        return repr_str

    def __call__(self):
        mask_per_frame = np.hstack(
            [
                np.zeros(self.num_patches_per_frame - self.num_masks_per_frame),
                np.ones(self.num_masks_per_frame),
            ]
        )
        mask_per_frame_list = []
        for _ in range(self.frames):
            np.random.shuffle(mask_per_frame)
            mask_per_frame_list.append(mask_per_frame)
        mask = np.array(mask_per_frame_list)
        print('RandomMasking!!!!')
        return mask
