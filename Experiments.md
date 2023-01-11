# Boostcamp AI Tech - Level2: CV-7 실험 기록


- Config file을 변경하지 않고 실험한다.
    - 단, lr scheduler는 20epoch에 맞춰서 학습한다. (bs = 16 안되면 8)
    - COCO, Pascal, ADE20k 에 대해서 Pretrain된 weight만 쓴다.
    - Aspect ratio가 1:1인 config file만을 사용 한다.
    - Backbone
        - ConvNext
        - SwinTransformer
        - HRNet
        - MAE
        
- 실험 결과 기록
    - pretrained weight가 뭔지 명시하기
    - confusion matrix 뽑아보기
    - Best epoch의 validation score, Leader board 제출 score 적기


## 12. 26
- AI stage 제출 description -> notion 실험 페이지 링크

## 12. 27
- 데일리스크럼 때 모델 선정 완료
    - 실험 설정은 configs/\_thlee00_/\_base_\upernet 안에 있음
    - base_upernet_convnext_xlarge_fp16_640x640_160k_ade20k.py
    - dataset, model, schedules 등 맞추기 헷갈리는 사람을 위한 base_whole_upernet~.py
        - img_norm 값 맞추고, aug 없이 기본 실험했던 설정 work_dir에서 가져온 것
- 주엽 미리 새로고침 중

## 12. 28 (F5 dAy~!~!~!~!)

- 주엽, 성수, 태희, 구 새로고침 중

## 12. 29 (F5 dAy~!~!~!~!)

- 성수, 구 새로고침 중

## 12. 30 (F5 dAy~!~!~!~!)

- 태희, 구, 기용 새로고침 중

## 12. 31 

- 주엽 떡국 끓이는 중 ..

## 01. 01 

- 주엽 떡국 먹는 중 ..

## 01. 02 


## 01. 03 

---

## 01. 04 


## 01. 05 
