import segmentation_models_pytorch as smp

class UnetPP_EfficientNetB0(smp.UnetPlusPlus):
    def __init__(self, num_classes=6):
        super().__init__(
            encoder_name="efficientnet-b0",   # энкодер EfficientNet-B0
            encoder_weights=None,             # или "imagenet", если обучал с предобученными весами
            in_channels=3,                    # вход RGB
            classes=num_classes               # число классов Potsdam (6)
        )
