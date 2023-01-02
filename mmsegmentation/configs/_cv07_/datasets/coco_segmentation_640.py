

# dataset settings
dataset_type = 'CustomDataset'
img_dir='/opt/ml/input/data/mmseg/img_dir/'
ann_dir= '/opt/ml/input/data/mmseg/ann_dir/'

# class settings
classes = ['Background', 'General trash', 'Paper', 'Paper pack', 'Metal', 'Glass', 'Plastic','Styrofoam', 'Plastic bag', 'Battery', 'Clothing']
palette =  [[0,0,0], [192,0,128], [0,128,192], [0,128,64], [128,0,0], [64,0,128],
           [64,0,192] ,[192,128,64], [192,192,128], [64,64,128], [128,0,192]]
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (640, 640)

albu_train_transforms = [
    dict(
    type='OneOf',
    transforms=[
        dict(type='ToGray', p=1),
        dict(type='Solarize', threshold=127, p=1),
        dict(type='Equalize', mode='cv', by_channels=True, mask=None, mask_params=(), always_apply=False, p=1),
        dict(type='RandomBrightnessContrast', brightness_limit=[0.1, 0.3], contrast_limit=[0.1, 0.3], p=1),
    ], p = 0.8
    ),
    # dict(type='UnsharpMask',blur_limit=(3, 7), sigma_limit=0.0, alpha=(0.2, 0.5), threshold=10, always_apply=False, p=0.5),
    dict(type='ShiftScaleRotate',shift_limit=0.0625, scale_limit=(0,1), rotate_limit=0, interpolation=1, p=0.5),
    dict(type='Cutout', num_holes=4, max_h_size=16, max_w_size=16),
]
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(640, 640),keep_ratio=True),
    # dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(
        type='Albu',
        transforms=albu_train_transforms,
        keymap=dict(img='image', gt_semantic_seg='mask'),
        update_pad_shape=True
    ),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
valid_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(640, 640),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        flip_direction=['horizontal', 'vertical'],
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
        img_scale=(640, 640),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        flip_direction=['horizontal', 'vertical'],
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
    samples_per_gpu=8,
    workers_per_gpu=4,
    # train=dict(
    #     classes=classes,
    #     palette=palette,
    #     type=dataset_type,
    #     reduce_zero_label=False, 
    #     img_dir=data_root + "images/train",
    #     ann_dir=data_root + "annotations/train",
    #     pipeline=train_pipeline),
    train=dict(
        type = 'MultiImageMixDataset',
        dataset = dict(
            type=dataset_type,
            classes=classes,
            img_dir=img_dir + 'train_fold0',
            ann_dir=ann_dir + 'train_fold0',
            pipeline=[dict(type='LoadImageFromFile'),
                      dict(type='LoadAnnotations')
            ]),
            pipeline=train_pipeline),
    val=dict(
        classes=classes,
        palette=palette,
        type=dataset_type,
        img_dir=img_dir + 'val_fold0',
        ann_dir=ann_dir + 'val_fold0',
        pipeline=valid_pipeline),
    test=dict(
        classes=classes,
        palette=palette,
        type=dataset_type,
        img_dir=img_dir+ 'test',
        pipeline=test_pipeline))