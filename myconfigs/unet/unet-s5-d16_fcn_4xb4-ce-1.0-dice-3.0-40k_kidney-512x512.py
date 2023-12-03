_base_ = [
    '../_base_/models/fcn_unet_s5-d16.py', '../_base_/datasets/kidney.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_2k.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(loss_decode=[
        dict(type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=1.0),
        dict(type='DiceLoss', loss_name='loss_dice', loss_weight=3.0)]),
    test_cfg=dict(crop_size=(512, 512), stride=(170, 170))
    )

