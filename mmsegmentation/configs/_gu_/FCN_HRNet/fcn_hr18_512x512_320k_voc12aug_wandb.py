_base_ = [
    '/opt/ml/input/code/mmsegmentation/configs/_gu_/_base_/models/fcn_hr18.py', '/opt/ml/input/code/mmsegmentation/configs/_gu_/_base_/datasets/coco_segmentation.py',
    '/opt/ml/input/code/mmsegmentation/configs/_gu_/_base_/runtime_fcn_hr48_wandb.py', '/opt/ml/input/code/mmsegmentation/configs/_gu_/_base_/schedules/schedule_20k.py'
]
model = dict(decode_head=dict(num_classes=11))
