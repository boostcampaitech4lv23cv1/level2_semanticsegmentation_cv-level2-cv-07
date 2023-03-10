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
<<<<<<< Updated upstream
load_from = '/opt/ml/input/level2_semanticsegmentation_cv-level2-cv-07/mmsegmentation/pretrained/beit_large_patch16_224_pt22k_ft22k_convert.pth'
=======
>>>>>>> Stashed changes
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
