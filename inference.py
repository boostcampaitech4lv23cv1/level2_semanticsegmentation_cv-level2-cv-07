from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
import os
import albumentations as A

import json

from dataset.dataset import *
from dataset.transforms import Transform
from model.models import *
from utils.setting import set_seed
from utils.collate import collate_fn
from utils.utils import get_result

def inference(model, test_loader, device):
    size = 256
    transform = A.Compose([A.Resize(size, size)])
    print('Start prediction.')
    
    model.eval()
    
    file_name_list = []
    preds_array = np.empty((0, size*size), dtype=np.long)
    
    with torch.no_grad():
        for step, (imgs, image_infos) in enumerate(tqdm(test_loader)):
            
            # inference (512 x 512)
            outs = model(torch.stack(imgs).to(device))['out']
            oms = torch.argmax(outs, dim=1).detach().cpu().numpy()
            
            # resize (256 x 256)
            temp_mask = []
            for img, mask in zip(np.stack(imgs), oms):
                transformed = transform(image=img, mask=mask)
                mask = transformed['mask']
                temp_mask.append(mask)
                
            oms = np.array(temp_mask)
            
            oms = oms.reshape([oms.shape[0], size*size]).astype(int)
            preds_array = np.vstack((preds_array, oms))
            
            file_name_list.append([i['file_name'] for i in image_infos])
    print("End prediction.")
    file_names = [y for x in file_name_list for y in x]
    
    return file_names, preds_array


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings(action='ignore')
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    cfg = json.load(open("cfg.json", "r"))
    
    batch_size = cfg["batch_size"]
    learning_rate = cfg["learning_rate"]
    num_epochs = cfg["epochs"]
    categories = cfg["categories"]
    seed = cfg["seed"]
    set_seed(seed)
    
    exp = cfg["exp"]
        
    transform = Transform()

    # Load test dataset
    test_path = os.path.join(cfg["data_dir"], cfg["ann_file"]["test"])
    test_dataset = CustomDataset(cfg["data_dir"], cfg["ann_file"]["train"], categories, mode='test', transform=transform.test)
    test_loader = DataLoader(dataset=test_dataset,
                                            batch_size=batch_size,
                                            num_workers=4,
                                            collate_fn=collate_fn)

    # 모델 호출
    model = torch.load(f"exp/{exp}/best.pt")

    # test set에 대한 prediction
    file_names, preds = inference(model, test_loader, device)

    get_result("output.csv")