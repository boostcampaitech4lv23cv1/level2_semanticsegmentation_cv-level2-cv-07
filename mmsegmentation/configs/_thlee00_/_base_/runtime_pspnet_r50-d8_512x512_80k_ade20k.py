# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook'),
        # dict(type='WandbLoggerHook',
        #      init_kwargs=dict(
        #          project='MMSeg',
        #          entity='CV07',
        #          reinit=True,
        #      ))
        # dict(type='PaviLoggerHook') # for internal services
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = '/opt/ml/input/level2_semanticsegmentation_cv-level2-cv-07/mmsegmentation/pretrained/pspnet_r50-d8_512x512_80k_ade20k_20200615_014128-15a8b914.pth'
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
