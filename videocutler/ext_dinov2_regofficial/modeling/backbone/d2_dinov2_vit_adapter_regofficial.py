from detectron2.layers import ShapeSpec
from detectron2.modeling import BACKBONE_REGISTRY, Backbone

from ext_dinov2_regofficial.vendor.vit_adapter_regofficial import ViTAdapterRegOfficial


@BACKBONE_REGISTRY.register()
class D2DinoV2ViTAdapterRegOfficial(Backbone):
    def __init__(self, cfg, input_shape):
        super().__init__()
        c = cfg.MODEL.DINO_ADAPTER_REGOFFICIAL
        self.net = ViTAdapterRegOfficial(
            img_size=c.IMG_SIZE,
            patch_size=c.PATCH_SIZE,
            embed_dim=c.EMBED_DIM,
            depth=c.DEPTH,
            num_heads=c.NUM_HEADS,
            mlp_ratio=c.MLP_RATIO,
            drop_path_rate=c.DROP_PATH_RATE,
            init_values=c.INIT_VALUES,
            num_register_tokens=c.NUM_REGISTER_TOKENS,
            interpolate_antialias=c.INTERPOLATE_ANTIALIAS,
            interpolate_offset=c.INTERPOLATE_OFFSET,
            conv_inplane=c.CONV_INPLANE,
            n_points=c.N_POINTS,
            deform_num_heads=c.DEFORM_NUM_HEADS,
            interaction_indexes=[list(x) for x in c.INTERACTION_INDEXES],
            with_cffn=c.WITH_CFFN,
            cffn_ratio=c.CFFN_RATIO,
            deform_ratio=c.DEFORM_RATIO,
            add_vit_feature=c.ADD_VIT_FEATURE,
            pretrained=c.PRETRAINED if c.PRETRAINED else None,
            use_extra_extractor=c.USE_EXTRA_EXTRACTOR,
            freeze_vit=c.FREEZE_VIT,
            strict_pretrain_load=c.STRICT_PRETRAIN_LOAD,
        )
        self._out_features = list(c.OUT_FEATURES)
        self._out_feature_strides = {"res2": 4, "res3": 8, "res4": 16, "res5": 32}
        self._out_feature_channels = {name: c.EMBED_DIM for name in self._out_features}

    def forward(self, x):
        feats = self.net(x)
        names = ["res2", "res3", "res4", "res5"]
        return {k: v for k, v in zip(names, feats) if k in self._out_features}

    def output_shape(self):
        return {
            name: ShapeSpec(channels=self._out_feature_channels[name], stride=self._out_feature_strides[name])
            for name in self._out_features
        }

    @property
    def size_divisibility(self):
        return 32
