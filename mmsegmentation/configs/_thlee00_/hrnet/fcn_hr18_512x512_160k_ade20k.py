_base_ = [
    '../_base_/models/fcn_hr18.py', '../_base_/datasets/coco_segmentation.py',
    '../_base_/runtime_fcn_hr18_512x512_160k_ade20k.py', '../_base_/schedules/schedule_160k.py'
]
model = dict(decode_head=dict(num_classes=150))