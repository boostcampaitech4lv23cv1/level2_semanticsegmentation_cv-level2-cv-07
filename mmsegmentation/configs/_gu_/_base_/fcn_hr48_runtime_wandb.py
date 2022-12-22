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
        dict(type='MMSegWandbHook',
         init_kwargs={'project': 'mmsegmentation'},
         interval=10,
         log_checkpoint=True,
         log_checkpoint_metadata=True,
         num_eval_images=100)
        # dict(type='PaviLoggerHook') # for internal services
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = '/opt/ml/input/code/mmsegmentation/pretrained/fcn_hr48_512x512_40k_voc12aug_20200613_222111-1b0f18bc.pth'
resume_from = None
workflow = [('train', 1), ('val', 1)]
cudnn_benchmark = True
cudnn_benchmark = True