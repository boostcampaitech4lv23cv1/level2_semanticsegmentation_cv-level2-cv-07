_base_ = [
    '../_base_/models/pspnet_r50-d8.py', '../_base_/datasets/coco_segmentation.py',
    '../_base_/runtime_pspnet_r50-d8_512x512_80k_ade20k.py', '../_base_/schedules/schedule_80k.py'
]
model = dict(
    decode_head=dict(num_classes=11), auxiliary_head=dict(num_classes=11))