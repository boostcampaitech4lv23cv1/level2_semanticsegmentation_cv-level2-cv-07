## Boostcamp AI Tech - Level2: CV-7

---

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

