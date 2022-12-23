from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO

import numpy as np
import cv2
import os

class CustomDataset(Dataset):
    """COCO format"""
    def __init__(self, data_dir, ann_file, categories, mode = 'train', transform = None):
        super().__init__()
        self.data_dir = data_dir
        self.categories = categories
        self.mode = mode
        self.transform = transform
        
        ann_path = os.path.join(data_dir, ann_file)
        self.coco = COCO(ann_path)
        
    def __getitem__(self, index: int):
        # dataset이 index되어 list처럼 동작
        image_id = self.coco.getImgIds(imgIds=index)
        image_infos = self.coco.loadImgs(image_id)[0]
        
        # cv2 를 활용하여 image 불러오기
        images = cv2.imread(os.path.join(self.data_dir, image_infos['file_name']))
        images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB).astype(np.float32)
        images /= 255.0
        
        if (self.mode in ('train', 'val')):
            ann_ids = self.coco.getAnnIds(imgIds=image_infos['id'])
            anns = self.coco.loadAnns(ann_ids)

            # Load the categories in a variable
            cat_ids = self.coco.getCatIds()
            cats = self.coco.loadCats(cat_ids)

            # masks : size가 (height x width)인 2D
            # 각각의 pixel 값에는 "category id" 할당
            # Background = 0
            masks = np.zeros((image_infos["height"], image_infos["width"]))
            # General trash = 1, ... , Cigarette = 10
            anns = sorted(anns, key=lambda idx : idx['area'], reverse=True)
            for i in range(len(anns)):
                className = self.get_classname(anns[i]['category_id'], cats)
                pixel_value = self.categories.index(className)
                masks[self.coco.annToMask(anns[i]) == 1] = pixel_value
            masks = masks.astype(np.int8)
                        
            # transform -> albumentations 라이브러리 활용
            if self.transform is not None:
                transformed = self.transform(image=images, mask=masks)
                images = transformed["image"]
                masks = transformed["mask"]
            return images, masks, image_infos
        
        if self.mode == 'test':
            # transform -> albumentations 라이브러리 활용
            if self.transform is not None:
                transformed = self.transform(image=images)
                images = transformed["image"]
            return images, image_infos
    
    def __len__(self) -> int:
        # 전체 dataset의 size를 return
        return len(self.coco.getImgIds())

    def get_classname(self, classID, cats):
        for i in range(len(cats)):
            if cats[i]['id']==classID:
                return cats[i]['name']
        return "None"


if __name__ == "__main__":
    import json
    import pandas as pd

    # collate_fn needs for batch
    def collate_fn(batch):
        return tuple(zip(*batch))

    cfg = json.load(open("../cfg.json", "r"))
    train = CustomDataset(data_dir=cfg["data_dir"], ann_file=cfg["ann_file"]["train"], categories = cfg["categories"], mode='train', transform=None)
    train_loader = DataLoader(dataset=train, 
                                        batch_size=8,
                                        shuffle=True,
                                        num_workers=4,
                                        collate_fn=collate_fn,
                                        drop_last=True)
    val = CustomDataset(data_dir=cfg["data_dir"], ann_file=cfg["ann_file"]["val"], categories = cfg["categories"], mode='val', transform=None)
    test = CustomDataset(data_dir=cfg["data_dir"], ann_file=cfg["ann_file"]["test"], categories = cfg["categories"], mode='test', transform=None)

    batch = next(iter(train_loader))
    for i, (image, mask, info) in enumerate(zip(*batch)):
        image*=255
        
        image = image.astype(np.uint8)

        mask = np.expand_dims(mask, axis=2)
        mask = mask.astype(np.uint8)

        color_map = np.array([
            [0, 0, 0],
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [255, 255, 0],
            [0, 255, 255],
            [255, 0, 255],
            [192, 128, 64],
            [192, 192, 128],
            [64, 64, 128],
            [128, 0, 192],
        ])

        ## segment의 bit 이미지 제작
        bit_mask = mask.copy()
        bit_mask[bit_mask>0] = 255
        bit_mask = cv2.cvtColor(bit_mask, cv2.COLOR_GRAY2RGB)

        ## 마스크, 세그먼트 원본, 배경 원본 작성
        mask = np.array(list(map(lambda x: color_map[x], mask)), dtype=np.uint8).squeeze()
        segment = cv2.bitwise_and(image, bit_mask)
        bg = cv2.subtract(image, bit_mask)

        masked_segment = cv2.addWeighted(segment, 0.5, mask, 0.5, 0)
        viz = cv2.bitwise_or(masked_segment, bg)
        
        cv2.imwrite(f"src{i}.jpg", viz)