_base_ = [
    '../_base_/models/pspnet_r50-d8.py', '../_base_/datasets/coco_segmentation.py',
    '../_base_/defalut_runtime.py', '../_base_/schedules/schedule_80k.py'
]
model = dict(
    decode_head=dict(num_classes=11), auxiliary_head=dict(num_classes=11))

load_from = '/opt/ml/input/level2_semanticsegmentation_cv-level2-cv-07/mmsegmentation/pretrained/pspnet_r50-d8_512x512_80k_ade20k_20200615_014128-15a8b914.pth'