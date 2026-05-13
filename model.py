import torch
import torch.nn as nn
import torchvision.models as models
import timm
import config


class NaanNeeModel(nn.Module):

    def __init__(self, num_classes=config.NUM_CLASSES):
        super(NaanNeeModel, self).__init__()

        # ── ResNet50 branch ────────────────────────────
        resnet = models.resnet50(
            weights=models.ResNet50_Weights.DEFAULT
        )
        self.resnet_features = nn.Sequential(
            *list(resnet.children())[:-1]
        )
        resnet_out_size = 2048

        # ── Vision Transformer branch ──────────────────
        self.vit = timm.create_model(
            "vit_small_patch16_224",
            pretrained=True,
            num_classes=0
        )
        vit_out_size = self.vit.num_features

        # ── Batch Normalization ────────────────────────
        combined_size = resnet_out_size + vit_out_size
        self.batch_norm = nn.BatchNorm1d(combined_size)

        # ── Fusion + Classifier ────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(combined_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # ResNet50 features
        resnet_out = self.resnet_features(x)
        resnet_out = resnet_out.view(resnet_out.size(0), -1)

        # ViT features
        vit_out = self.vit(x)

        # Concatenate both
        combined = torch.cat((resnet_out, vit_out), dim=1)

        # Batch normalization
        combined = self.batch_norm(combined)

        # Final prediction
        out = self.classifier(combined)
        return out


def get_model(num_classes=None):
    if num_classes is None:
        num_classes = config.NUM_CLASSES
    model = NaanNeeModel(num_classes=num_classes)
    return model
