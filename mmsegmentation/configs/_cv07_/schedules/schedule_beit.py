# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)

# Epoch based learner
runner = dict(type="EpochBasedRunner", max_epochs=30)
checkpoint_config = dict(interval=1,max_keep_ckpts=5)
evaluation = dict(interval=1, metric="mIoU", save_best='mIoU',pre_eval=True)