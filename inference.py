from torch.utils.data import DataLoader
import torch
from tqdm import tqdm

import json

from dataset.dataset import *
from model.models import *
from utils.setting import set_seed

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
            outs = model(torch.stack(imgs).to(device))
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
    set_seed(21)
    
    
    # Load test dataset
    test_dataset = CustomDataLoader(data_dir=test_path, mode='test', transform=test_transform)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=batch_size,
                                            num_workers=4,
                                            collate_fn=collate_fn)

    # test set에 대한 prediction
    file_names, preds = inference(model, test_loader, device)

    # PredictionString 대입
    for file_name, string in zip(file_names, preds):
        submission = submission.append({"image_id" : file_name, "PredictionString" : ' '.join(str(e) for e in string.tolist())}, 
                                    ignore_index=True)

    # sample_submisson.csv 열기
    submission = pd.read_csv('./submission/sample_submission.csv', index_col=None)

    # submission.csv로 저장
    submission.to_csv("./submission/Unet_best_model.csv", index=False)