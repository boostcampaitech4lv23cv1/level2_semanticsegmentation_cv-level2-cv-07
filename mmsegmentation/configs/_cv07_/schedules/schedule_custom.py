# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=6550)
checkpoint_config = dict(by_epoch=False, interval=7000)
evaluation = dict(interval=327, metric='mIoU', save_best='mIoU',pre_eval=True)

# Epoch based learner
runner = dict(type="EpochBasedRunner", max_epochs=20)
checkpoint_config = dict(interval=21)
evaluation = dict(interval=1, metric="mIoU", save_best='mIoU',pre_eval=True)