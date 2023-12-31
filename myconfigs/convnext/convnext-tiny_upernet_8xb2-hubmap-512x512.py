_base_ = [
    '../_base_/models/upernet_convnext.py', '../_base_/datasets/hubmap.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-tiny_3rdparty_32xb128-noema_in1k_20220301-795e9634.pth'  # noqa
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='mmpretrain.ConvNeXt',
        arch='tiny',
        out_indices=[0, 1, 2, 3],
        drop_path_rate=0.4,
        layer_scale_init_value=1.0,
        gap_before_final_norm=False,
        init_cfg=dict(
            type='Pretrained', checkpoint=checkpoint_file,
            prefix='backbone.')),
    decode_head=dict(
        in_channels=[96, 192, 384, 768],
        num_classes=3,
        loss_decode=[
            dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4, class_weight=[0.25, 1, 0.5]),
            dict(
                type='FocalLoss', use_sigmoid=True, loss_weight=1.0, gamma=2.0, alpha=[0.25, 1.0, 0.5])]
    ),
    auxiliary_head=dict(in_channels=384, num_classes=3,
                        loss_decode=[
                            dict(
                                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4,
                                class_weight=[0.25, 1, 0.5]),
                            dict(
                                type='FocalLoss', use_sigmoid=True, loss_weight=1.0, gamma=2.0, alpha=[0.25, 1, 0.5])]
                        ),
    test_cfg=dict(mode='whole'),
)

optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05),
    paramwise_cfg={
        'decay_rate': 0.9,
        'decay_type': 'stage_wise',
        'num_layers': 6
    },
    constructor='LearningRateDecayOptimizerConstructor',
    loss_scale='dynamic')

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        power=1.0,
        begin=1500,
        end=160000,
        eta_min=0.0,
        by_epoch=False,
    )
]

# By default, models are trained on 8 GPUs with 2 images per GPU
train_dataloader = dict(batch_size=4)
val_dataloader = dict(batch_size=4)
test_dataloader = val_dataloader
