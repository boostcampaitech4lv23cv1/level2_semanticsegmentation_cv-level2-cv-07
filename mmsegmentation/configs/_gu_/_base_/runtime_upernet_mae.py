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
load_from = '/opt/ml/input/code/mmsegmentation/pretrained/upernet_mae-base_fp16_8x2_512x512_160k_ade20k_20220426_174752-f92a2975.pth'
resume_from = None
workflow = [('train', 1), ('val', 1)]
cudnn_benchmark = True