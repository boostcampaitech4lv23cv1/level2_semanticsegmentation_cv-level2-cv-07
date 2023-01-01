# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=140000)
checkpoint_config = dict(by_epoch=False, interval=1000000)
evaluation = dict(interval=1309, metric='mIoU', save_best='mIoU',pre_eval=True)
# Epoch based learner
runner = dict(type="EpochBasedRunner", max_epochs=100)
checkpoint_config = dict(interval=101)
evaluation = dict(interval=1, metric="mIoU", save_best='mIoU',pre_eval=True)