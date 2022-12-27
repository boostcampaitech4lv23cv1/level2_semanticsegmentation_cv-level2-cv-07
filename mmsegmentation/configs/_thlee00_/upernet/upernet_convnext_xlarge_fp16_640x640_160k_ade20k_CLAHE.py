_base_ = [
    '../_base_/models/upernet_convnext.py',
    #'../_base_/datasets/coco_segmentation_640_aug.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_custom.py'
]
crop_size = (640, 640)
checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-xlarge_3rdparty_in21k_20220301-08aa5ddc.pth'  # noqa
model = dict(
    backbone=dict(
        type='mmcls.ConvNeXt',
        arch='xlarge',
        out_indices=[0, 1, 2, 3],
        drop_path_rate=0.4,
        layer_scale_init_value=1.0,
        gap_before_final_norm=False,
        init_cfg=dict(
            type='Pretrained', checkpoint=checkpoint_file,
            prefix='backbone.')),
    decode_head=dict(
        in_channels=[256, 512, 1024, 2048],
        num_classes=11,
    ),
    auxiliary_head=dict(in_channels=1024, num_classes=11),
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(426, 426)),
)

optimizer = dict(
    constructor='LearningRateDecayOptimizerConstructor',
    _delete_=True,
    type='AdamW',
    lr=0.00008,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg={
        'decay_rate': 0.9,
        'decay_type': 'stage_wise',
        'num_layers': 12
    })

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)

load_from = '/opt/ml/input/level2_semanticsegmentation_cv-level2-cv-07/mmsegmentation/pretrained/upernet_convnext_xlarge_fp16_640x640_160k_ade20k_20220226_080344-95fc38c2.pth'

# By default, models are trained on 8 GPUs with 2 images per GPU
data = dict(samples_per_gpu=8)
# fp16 settings
optimizer_config = dict(type='Fp16OptimizerHook', loss_scale='dynamic')
# fp16 placeholder
fp16 = dict()


# dataset settings
dataset_type = 'CustomDataset'
img_dir='/opt/ml/input/data/mmseg/img_dir/'
ann_dir= '/opt/ml/input/data/mmseg/ann_dir/'

classes = ("Backgroud","General trash", "Paper", "Paper pack", "Metal", "Glass", 
            "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")
palette =  [[0,0,0], [192,0,128], [0,128,192], [0,128,64], [128,0,0], [64,0,128],
           [64,0,192] ,[192,128,64], [192,192,128], [64,64,128], [128,0,192]]

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# train_all mean, std


train_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations'),
        dict(type='Resize', img_scale=(640,640), keep_ratio=True),
        #dict(type='PhotoMetricDistortion'),
        dict(type='CLAHE', clip_limit=40.0, tile_grid_size=(8, 8)),
        dict(type='RandomFlip', flip_ratio=0.5),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='Pad', size_divisor=32),
        #dict(type='RandomMosaic', prob=0.5, img_scale=(640,640)),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_semantic_seg']),
    ]

val_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(
            type='MultiScaleFlipAug',
            img_scale=(640, 640),
            flip=False,
            transforms=[
                dict(type='Resize', keep_ratio=True),
                dict(type='RandomFlip'),
                dict(type='Normalize', **img_norm_cfg),
                dict(type='Pad', size_divisor=32),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img']),
            ])
    ]

test_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(
            type='MultiScaleFlipAug',
            img_scale=[(640,640)],#[(1024, 1024),(512,512),(1333,800)],
            flip= False,
            flip_direction =  ["horizontal", "vertical" ],
            transforms=[
                dict(type='Resize', keep_ratio=True),
                dict(type='RandomFlip'),
                dict(type='Normalize', **img_norm_cfg),
                dict(type='Pad', size_divisor=32),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img']),
            ])
    ]
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_dir=ann_dir + 'train',
        img_dir=img_dir + 'train',
        classes = classes,
        palette= palette,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_dir=ann_dir + 'val',
        img_dir=img_dir + 'val',
        classes = classes,
        palette= palette,
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        img_dir=img_dir+'test' ,
        classes = classes,
        palette= palette,
        pipeline=test_pipeline))