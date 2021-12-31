from .big_transfer import models as bit_models
import torch
from pathlib import Path
import faiss

def load_model(saved_path, num_classes):
    
    model_ft = bit_models.KNOWN_MODELS['BiT-M-R50x1'](head_size=num_classes, zero_head=True)
    weights = torch.load(saved_path)
    model_ft.load_state_dict(weights)
    
    model_ft = model_ft.to('cuda:0')
    model_ft.eval()
    return model_ft


def emb_person(absolute_repo_path, drone_work_path, person_imgs):
    
    # 요구조자인지 체크
    saved_path = str(Path(absolute_repo_path) /'weights/task_03/bit_rescue/R50x1_224.pt')
    model = load_model(saved_path, 2)
    infer_cls(model, person_imgs)
    del model
    
    # 임베딩
    saved_path = str(Path(absolute_repo_path) /'weights/task_03/bit_emb/R50x1_224.pt')
    model = load_model(saved_path, 512)
    infer_cls(model, person_imgs)
    del model
    
    
    
def count_person(absolute_repo_path, set_work_path, set_person_imgs, set_person_embs):
    
    # 동일 비디오에서 움직임이 있는지 체크 
    saved_path = str(Path(absolute_repo_path) /'weights/task_03/bit_action/R50x1_224.pt')
    model = load_model(saved_path, 2)
    infer_cls(model, set_person_imgs)
    del model
    
    # 다른 비디오에서 장소 이동이 있는지 체크
    saved_path = str(Path(absolute_repo_path) /'weights/task_03/bit_place/R50x1_224.pt')
    model = load_model(saved_path, 2)
    infer_cls(model, set_person_imgs)
    del model
    
    return 
    

#########################################################
from glob import glob

from PIL import Image
from torchvision import datasets, models, transforms
import numpy as np

trsf =  transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
       

def infer_cls(model, img_list):
    model = model.to('cuda:0')
    img_list = glob('/home/agc2021/temp/task_03/set_1/drone_1/person/*.jpg')
    
    total_outs = []
    with torch.no_grad():
        for e in img_list:
            im = Image.open(e)
            inputs = trsf(im)
            inputs = inputs.unsqueeze(0)
            outs = model(inputs.to('cuda:0'))
            total_outs.append(outs.cpu().numpy())
    return np.stack(total_outs)
    