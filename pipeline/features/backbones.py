"""Frozen-backbone registry. key -> timm weight name (+ optional create_model kwargs).
Each entry records its pretraining tier."""

BACKBONES = {
    "resnet18":       dict(name="resnet18.a1_in1k"),
    "mobilenetv3_l":  dict(name="mobilenetv3_large_100.ra_in1k"),
    "effnet_b0_ns":   dict(name="tf_efficientnet_b0.ns_jft_in1k"),
    "convnext_t_22k": dict(name="convnext_tiny.fb_in22k"),
    "vit_s16_21k":    dict(name="vit_small_patch16_224.augreg_in21k"),
    "dinov2_vits14":  dict(name="vit_small_patch14_dinov2.lvd142m", model_kwargs=dict(img_size=224)),
}
