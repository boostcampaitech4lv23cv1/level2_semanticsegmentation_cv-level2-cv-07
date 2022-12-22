import json
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os

import random
import torch.backends.cudnn as cudnn

from dataset.dataset import *
from dataset.transforms import Transform
from model.models import *
from utils.utils import *
from utils.setting import set_seed
from utils.preprocess import exp_generator
from utils.collate import collate_fn

def validation(epoch, model, data_loader, criterion, categories, device):
    print(f'Start validation #{epoch}')
    model.eval()

    with torch.no_grad():
        n_class = 11
        total_loss = 0
        cnt = 0
        
        hist = np.zeros((n_class, n_class))
        for step, (images, masks, _) in enumerate(data_loader):
            
            images = torch.stack(images)       
            masks = torch.stack(masks).long()  

            images, masks = images.to(device), masks.to(device)            
            
            # device 할당
            model = model.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss
            cnt += 1
            
            outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            masks = masks.detach().cpu().numpy()
            
            hist = add_hist(hist, masks, outputs, n_class=n_class)
        
        acc, acc_cls, mIoU, fwavacc, IoU = label_accuracy_score(hist)
        IoU_by_class = [{classes : round(IoU,4)} for IoU, classes in zip(IoU , categories)]
        
        avrg_loss = total_loss / cnt
        print(f'Validation #{epoch}  Average Loss: {round(avrg_loss.item(), 4)}, Accuracy : {round(acc, 4)}, \
                mIoU: {round(mIoU, 4)}')
        print(f'IoU by class : {IoU_by_class}')
        
    return avrg_loss

def train(num_epochs, model, data_loader, val_loader, criterion, optimizer, exp, categories, device):
    print(f'Start training..')
    n_class = 11
    best_loss = 9999999
    
    for epoch in range(num_epochs):
        model.train()

        hist = np.zeros((n_class, n_class))
        for step, (images, masks, _) in enumerate(data_loader):
            images = torch.stack(images)       
            masks = torch.stack(masks).long() 
            
            # gpu 연산을 위해 device 할당
            images, masks = images.to(device), masks.to(device)
            
            # device 할당
            model = model.to(device)
            
            # inference
            outputs = model(images)

            # loss 계산 (cross entropy loss)
            loss = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            masks = masks.detach().cpu().numpy()
            
            hist = add_hist(hist, masks, outputs, n_class=n_class)
            acc, acc_cls, mIoU, fwavacc, IoU = label_accuracy_score(hist)
            
            # step 주기에 따른 loss 출력
            if (step + 1) % 25 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{step+1}/{len(train_loader)}], \
                        Loss: {round(loss.item(),4)}, mIoU: {round(mIoU,4)}')
             
        # validation 주기에 따른 loss 출력 및 best model 저장
        avrg_loss = validation(epoch + 1, model, val_loader, criterion, categories, device)

        # 최근과 최고 epoch에 대한 last.pt와 best.pt 생성
        torch.save(model, f"exp/{exp}/last.pt")
        if avrg_loss < best_loss:
            print(f"Best performance at epoch: {epoch + 1}")
            print(f"Save model in exp/{exp}")
            best_loss = avrg_loss
            torch.save(model, f"exp/{exp}/best.pt")
            best_model=model
        
        if scheduler is not None:
            scheduler.step()

    return best_model


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

    # import pprint
    # pprint.pprint(timm.models.list_models())

    transform = Transform()

    ## Load train dataset
    train_dataset = CustomDataset(cfg["data_dir"], cfg["ann_file"]["train"], categories, mode='train', transform=transform.train)
    train_loader = DataLoader(dataset=train_dataset, 
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=4,
                                            collate_fn=collate_fn,
                                            drop_last=True)


    ## Load validation dataset
    val_dataset = CustomDataset(cfg["data_dir"], cfg["ann_file"]["train"], categories, mode='val', transform=transform.val)
    val_loader = DataLoader(dataset=val_dataset, 
                                            batch_size=batch_size,
                                            shuffle=False,
                                            num_workers=4,
                                            collate_fn=collate_fn)


    exp = exp_generator()
    scheduler = None

    model = CustomModel()
    model = model.to(device)

    # Loss function 정의
    criterion = nn.CrossEntropyLoss()

    # Optimizer 정의
    optimizer = torch.optim.Adam(params = model.parameters(), lr = learning_rate, weight_decay=1e-6)

    train(num_epochs, model, train_loader, val_loader, criterion, optimizer, exp, categories, device)
