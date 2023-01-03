# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook'),
        # dict(type='MMSegWandbHook',
        #  init_kwargs={'project': 'mmsegmentation'},
        #  interval=10,
        #  log_checkpoint=True,
        #  log_checkpoint_metadata=True,
        #  num_eval_images=50)
        # dict(type='PaviLoggerHook') # for internal services
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
resume_from = None
workflow = [('train', 1), ('val', 1)]
cudnn_benchmark = True
