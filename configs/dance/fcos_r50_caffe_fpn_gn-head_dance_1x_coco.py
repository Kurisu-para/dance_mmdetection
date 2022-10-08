_base_ = [
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
# model settings
model = dict(
    type='Dance',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='caffe',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://detectron/resnet50_caffe')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        #注释掉start_level(start_level = 0), 这样会多拿到一层P2的特征，用于edge_det的实现
        start_level= 0,
        add_extra_convs='on_output',  # use P5
        #相应的num_outs应该变为6
        num_outs=6,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='FCOSDanceHead',
        #type='FCOSHead',
        num_classes=80,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        #strides=[4, 8, 16, 32, 64, 128]
        #edge_on=True
        strides=[8, 16, 32, 64, 128],
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='IoULoss', loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_ext=dict(
                    type='SmoothL1BoxLoss', loss_weight=1.0)),
    mask_head=dict(
        type='RefineHead',
        num_classes=80,
        in_channels=256,
        feat_channels=256,
        #strides=[4, 8, 16, 32, 64, 128]
        #edge_on=True
        strides=[4, 8, 16, 32],
        num_iter = (0, 0, 1),
        num_convs = 2,
        num_sampling = 196,
        in_features = ['p2', 'p3', 'p4', 'p5'],
        common_stride=4,
        loss_refine=dict(
                    type='SmoothL1Loss', loss_weight=10.0),
        loss_edge=dict(type='DiceIgnoreLoss', use_sigmoid = True, activate = False, loss_weight=1.0),
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
    ),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100)
    )
img_norm_cfg = dict(
    mean=[102.9801, 115.9465, 122.7717], std=[1.0, 1.0, 1.0], to_rgb=False)
#img_norm_cfg = dict(
#    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
#as is used in coco_instance
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True, with_seg=True, poly2mask=False),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks', 'gt_semantic_seg'],
                            meta_keys=('filename', 'ori_filename', 'ori_shape', 'img_shape')),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'],
                                meta_keys=('filename', 'ori_filename', 'ori_shape', 'img_shape', 'scale_factor')),
        ])
]

dataset_type = 'CocoDataset'
#generated edge coco
data_root = '/home/sjtu/scratch/shaoyinkang/dance/datasets/coco/'

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
            type=dataset_type,
            ann_file=data_root + 'annotations/instances_train2017.json',
            img_prefix=data_root + 'train2017/',
            seg_prefix=data_root + 'edge_train2017/',
            pipeline=train_pipeline),
    val=dict(
            type=dataset_type,
            ann_file=data_root + 'annotations/instances_val2017.json',
            img_prefix=data_root + 'val2017/',
            pipeline=test_pipeline),
    test=dict(
            type=dataset_type,
            ann_file=data_root + 'annotations/instances_val2017.json',
            img_prefix=data_root + 'val2017/',
            pipeline=test_pipeline))
evaluation = dict(interval=1, metric=['bbox', 'segm'])
# optimizer
#optimizer = dict(
#    lr=0.01, paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.))
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
# here we use the same policy as dance in detectron2
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 1000,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)
find_unused_parameters=True