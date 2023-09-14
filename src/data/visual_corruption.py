import os, random, cv2
import albumentations as A
import numpy as np
import torchvision
import torch
from skimage.util import random_noise

class Visual_Corruption_Modeling:
    def __init__(self, d_image='./occlusion_patch/object_image_sr', d_mask='./occlusion_patch/object_mask_x4'):
        assert os.path.exists(d_image), "Please download coco_object.7z first"
        self.d_image = d_image
        self.d_mask = d_mask
        self.aug = get_occluder_augmentor()
        self.occlude_imgs = os.listdir(d_image)

    def get_occluders(self):
        occlude_img = random.choice(self.occlude_imgs)

        occlude_mask = occlude_img.replace('jpeg', 'png')

        ori_occluder_img = cv2.imread(os.path.join(self.d_image, occlude_img), -1)
        ori_occluder_img = cv2.cvtColor(ori_occluder_img, cv2.COLOR_BGR2RGB)

        occluder_mask = cv2.imread(os.path.join(self.d_mask, occlude_mask))
        occluder_mask = cv2.cvtColor(occluder_mask, cv2.COLOR_BGR2GRAY)

        occluder_mask = cv2.resize(occluder_mask, (ori_occluder_img.shape[1], ori_occluder_img.shape[0]),
                                    interpolation=cv2.INTER_LANCZOS4)

        occluder_img = cv2.bitwise_and(ori_occluder_img, ori_occluder_img, mask=occluder_mask)

        transformed = self.aug(image=occluder_img, mask=occluder_mask)
        occluder_img, occluder_mask = transformed["image"], transformed["mask"]

        occluder_size = random.choice(range(20, 46))

        occluder_img = cv2.resize(occluder_img, (occluder_size, occluder_size), interpolation= cv2.INTER_LANCZOS4)
        occluder_mask = cv2.resize(occluder_mask, (occluder_size, occluder_size), interpolation= cv2.INTER_LANCZOS4)

        return occlude_img, occluder_img, occluder_mask

    def noise_sequence(self, img_seq, freq=1):
        if freq == 1:
            len = img_seq.shape[0]
            occ_len = random.randint(int(len * 0.1), int(len * 0.5))
            start_fr = random.randint(0, len-occ_len)

            raw_sequence = img_seq[start_fr:start_fr+occ_len]
            prob = random.random()
            if prob < 0.3:
                var = random.random() * 0.2
                raw_sequence = np.expand_dims(raw_sequence, 3)
                raw_sequence = random_noise(raw_sequence, mode='gaussian', mean=0, var=var, clip=True) * 255
                raw_sequence = raw_sequence.squeeze(3)
            elif prob < 0.6:
                blur = torchvision.transforms.GaussianBlur(kernel_size=(7, 7), sigma=(0.1, 2.0))
                raw_sequence = np.expand_dims(raw_sequence, 3)
                raw_sequence = blur(torch.tensor(raw_sequence).permute(0, 3, 1, 2)).permute(0, 2, 3, 1).numpy()
                raw_sequence = raw_sequence.squeeze(3)
            else:
                pass

            img_seq[start_fr:start_fr + occ_len] = raw_sequence

        else:
            len_global = img_seq.shape[0]
            len = img_seq.shape[0] // freq
            for j in range(freq):
                try:
                    occ_len = random.randint(int(len_global * 0.3), int(len_global * 0.5))
                    start_fr = random.randint(0, len*j + len - occ_len)
                    if start_fr < len*j:
                        assert 1==2
                except:
                    occ_len = len // 2
                    start_fr = len * j

                raw_sequence = img_seq[start_fr:start_fr + occ_len]
                prob = random.random()
                if prob < 0.3:
                    var = random.random() * 0.2
                    raw_sequence = np.expand_dims(raw_sequence, 3)
                    raw_sequence = random_noise(raw_sequence, mode='gaussian', mean=0, var=var, clip=True) * 255
                    raw_sequence = raw_sequence.squeeze(3)
                elif prob < 0.6:
                    blur = torchvision.transforms.GaussianBlur(kernel_size=(7, 7), sigma=(0.1, 2.0))
                    raw_sequence = np.expand_dims(raw_sequence, 3)
                    raw_sequence = blur(torch.tensor(raw_sequence).permute(0, 3, 1, 2)).permute(0, 2, 3, 1).numpy()
                    raw_sequence = raw_sequence.squeeze(3)
                else:
                    pass

                img_seq[start_fr:start_fr + occ_len] = raw_sequence

        return img_seq


    def occlude_sequence(self, img_seq, landmarks, yx_min, freq=1):
        if freq == 1:
            occlude_img, occluder_img, occluder_mask = self.get_occluders()

            len = img_seq.shape[0]
            start_pt_idx = random.randint(48,67)
            offset = random.randint(20,60)
            occ_len = random.randint(int(len * 0.1), int(len * 0.5))
            start_fr = random.randint(0, len-occ_len)

            for i in range(occ_len):
                fr = cv2.cvtColor(img_seq[i+start_fr], cv2.COLOR_GRAY2RGB)
                x, y = landmarks[i][start_pt_idx]

                alpha_mask = np.expand_dims(occluder_mask, axis=2)
                alpha_mask = np.repeat(alpha_mask, 3, axis=2) / 255.0

                fr = self.overlay_image_alpha(fr, occluder_img, int(y-yx_min[i][0]-offset), int(x-yx_min[i][1]-offset), alpha_mask)
                img_seq[i + start_fr] = cv2.cvtColor(fr, cv2.COLOR_RGB2GRAY)

        else:
            len_global = img_seq.shape[0]
            len = img_seq.shape[0] // freq
            for j in range(freq):
                occlude_img, occluder_img, occluder_mask = self.get_occluders()

                start_pt_idx = random.randint(48, 67)
                offset = random.randint(20, 40)
                try:
                    occ_len = random.randint(int(len_global * 0.3), int(len_global * 0.5))
                    start_fr = random.randint(0, len*j + len - occ_len)
                    if start_fr < len*j:
                        assert 1==2
                except:
                    occ_len = len // 2
                    start_fr = len * j

                for i in range(occ_len):
                    fr = cv2.cvtColor(img_seq[i + start_fr], cv2.COLOR_GRAY2RGB)
                    x, y = landmarks[i][start_pt_idx]

                    alpha_mask = np.expand_dims(occluder_mask, axis=2)
                    alpha_mask = np.repeat(alpha_mask, 3, axis=2) / 255.0
                    fr = self.overlay_image_alpha(fr, occluder_img, int(y - yx_min[i][0] - offset), int(x - yx_min[i][1] - offset),
                                                alpha_mask)
                    img_seq[i + start_fr] = cv2.cvtColor(fr, cv2.COLOR_RGB2GRAY)
        return img_seq, occlude_img

    def overlay_image_alpha(self, img, img_overlay, x, y, alpha_mask):
        """Overlay `img_overlay` onto `img` at (x, y) and blend using `alpha_mask`.

        `alpha_mask` must have same HxW as `img_overlay` and values in range [0, 1].
        """
        # Image ranges
        y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
        x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

        # Overlay ranges
        y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
        x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

        # Exit if nothing to do
        if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
            return

        # Blend overlay within the determined ranges
        img_crop = img[y1:y2, x1:x2]
        img_overlay_crop = img_overlay[y1o:y2o, x1o:x2o]

        alpha = alpha_mask[y1o:y2o, x1o:x2o]
        alpha_inv = 1.0 - alpha
        img_crop[:] = alpha * img_overlay_crop + alpha_inv * img_crop
        return img

def get_occluder_augmentor():
    """
    Occludor augmentor
    """
    aug=A.Compose([
        A.AdvancedBlur(),
        A.OneOf([
            A.ImageCompression (quality_lower=70,p=0.5),
            ], p=0.5),
        A.Affine  (
            scale=(0.8,1.2),
            rotate=(-15,15),
            shear=(-8,8),
            fit_output=True,
            p=0.7
        ),
        A.RandomBrightnessContrast(p=0.5,brightness_limit=0.1, contrast_limit=0.1, brightness_by_max=False),
        ])
    return aug
