_base_ = './fcn_hr18_480x480_80k_pascal_context.py'
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

load_from = '/opt/ml/input/level2_semanticsegmentation_cv-level2-cv-07/mmsegmentation/pretrained/fcn_hr48_480x480_80k_pascal_context_20200911_155322-847a6711.pth'

# runtime settings
runner = dict(type='IterBasedRunner', max_iters=3300)
checkpoint_config = dict(by_epoch=False, interval=165)
evaluation = dict(interval=165, metric='mIoU', save_best='mIoU', pre_eval=True)