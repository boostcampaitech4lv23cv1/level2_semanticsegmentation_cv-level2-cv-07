_base_ = './fcn_hr18_512x512_160k_ade20k.py'
data = dict(samples_per_gpu=16)
model = dict(
    pretrained='open-mmlab://msra/hrnetv2_w48',
    backbone=dict(
        extra=dict(
            stage2=dict(num_channels=(48, 96)),
            stage3=dict(num_channels=(48, 96, 192)),
            stage4=dict(num_channels=(48, 96, 192, 384)))),
    decode_head=dict(
        in_channels=[48, 96, 192, 384], channels=sum([48, 96, 192, 384]),
        num_classes=11))

load_from = '/opt/ml/input/level2_semanticsegmentation_cv-level2-cv-07/mmsegmentation/pretrained/fcn_hr48_512x512_160k_ade20k_20200614_214407-a52fc02c.pth'

# runtime settings
runner = dict(type='IterBasedRunner', max_iters=3300)
checkpoint_config = dict(by_epoch=False, interval=165)
evaluation = dict(interval=165, metric='mIoU', save_best='mIoU', pre_eval=True)