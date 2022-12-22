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
load_from = '/opt/ml/input/level2_semanticsegmentation_cv-level2-cv-07/mmsegmentation/pretrained/fcn_hr48_512x512_160k_ade20k_20200614_214407-a52fc02c.pth'
resume_from = None
workflow = [('train', 1), ('val', 1)]
cudnn_benchmark = True
