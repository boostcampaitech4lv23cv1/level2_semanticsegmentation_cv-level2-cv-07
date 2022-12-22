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
    cfg = json.load(open("../cfg.json", "r"))
    train = CustomDataset(data_dir="../../data", ann_path="train.json", categories = cfg["categories"], mode='train', transform=None)
    val = CustomDataset(data_dir="../../data", ann_path="val.json", categories = cfg["categories"], mode='val', transform=None)
    val = CustomDataset(data_dir="../../data", ann_path="test.json", categories = cfg["categories"], mode='test', transform=None)

    print(type(train[0]))