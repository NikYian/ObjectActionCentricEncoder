import torch
import torch.nn as nn
import clip
import sys
import torch.nn as nn

sys.path.append(r"./externals/hopfield-layers")
sys.path.append(r"./externals/mae")
from hflayers import HopfieldLayer, Hopfield
import models_mae


def prepare_model(chkpt_dir, arch="mae_vit_base_patch16"):
    # build model
    model = getattr(models_mae, arch)()
    # load model
    checkpoint = torch.load(chkpt_dir, map_location="cpu")
    msg = model.load_state_dict(checkpoint["model"], strict=False)
    print(msg)
    return model


class ClassificationHead(nn.Module):
    def __init__(self, input_size, num_classes=10, nn_type="MLP"):
        super(ClassificationHead, self).__init__()
        if nn_type == "MLP":
            self.classifiers = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.ReLU(inplace=False), nn.Linear(input_size, 1), nn.Sigmoid()
                    )
                    for _ in range(num_classes)
                ]
            )
        else:
            self.classifiers = nn.ModuleList(
                [
                    nn.Sequential(
                        HopfieldLayer(
                            input_size=input_size,
                            quantity=100,
                            num_heads=1,
                            hidden_size=input_size,
                            stored_pattern_size=input_size,
                            pattern_projection_size=input_size,
                            lookup_weights_as_separated=True,
                            output_size=1,
                            scaling=0.2,
                            lookup_targets_as_trainable=True,
                            stored_pattern_as_static=True,
                            state_pattern_as_static=True,
                        ),
                        nn.Sigmoid(),
                    )
                    for _ in range(num_classes)
                ]
            )

    def forward(self, x):
        outputs = [classifier(x) for classifier in self.classifiers]
        return torch.cat(outputs, dim=-1)


class AcEnn(nn.Module):
    # Action Centric Encoder Module

    def __init__(
        self,
        args,
        head="MLP",
        image_features="mae",
        ACM_features="combo",
    ):
        super(AcEnn, self).__init__()
        self.ACM_features = ACM_features
        self.image_features = image_features
        self.args = args
        self.thresholds = torch.tensor([0.5] * 10).to(args.device)

        for param in self.clip.parameters():  # CLIP params are frozen
            param.requires_grad = False
        self.head_type = head
        if image_features == "clip":
            self.image_encoder, self.preprocess = clip.load(
                args.CLIP_model, device=args.device
            )
            self.image_features_dim = self.image_encoder.visual.output_dim
            self.encode_image = self.image_encoder.encode_image
        elif image_features == "mae":
            chkpt_dir = "/gpu-data2/nyian/chackpoints/mae_vit_base_patch16.pth"
            self.image_encoder = prepare_model(chkpt_dir, "mae_vit_base_patch16").to(
                args.device
            )
            self.image_features_dim = 768  # MAE features
            self.encode_image = self.encode_mae

        if head == "MLP":
            layers = []
            layers.append(nn.Linear(self.image_features_dim, args.AcE_hidden_layers[0]))
            layers.append(nn.ReLU(inplace=False))
            layers.append(nn.Dropout(args.AcE_dropout_rate))

            for i in range(len(args.AcE_hidden_layers) - 1):
                layers.append(
                    nn.Linear(args.AcE_hidden_layers[i], args.AcE_hidden_layers[i + 1])
                )
                layers.append(nn.ReLU(inplace=False))
                layers.append(nn.Dropout(args.AcE_dropout_rate))

            layers.append(nn.Linear(args.AcE_hidden_layers[-1], args.AcE_feature_size))
            self.head = nn.Sequential(*layers)
        elif head == "Hopfield":
            self.head = HopfieldLayer(
                input_size=self.image_features_dim,
                quantity=1000,
                num_heads=10,
                hidden_size=self.image_features_dim,
                lookup_weights_as_separated=True,
                stored_pattern_size=self.image_features_dim,
                pattern_projection_size=self.image_features_dim,
                output_size=args.AcE_feature_size,
                scaling=0.1,
                lookup_targets_as_trainable=True,
                stored_pattern_as_static=True,
                state_pattern_as_static=True,
                dropout=args.AcE_dropout_rate,
            )

        if ACM_features == "image":
            self.classification_head = ClassificationHead(
                input_size=self.image_features_dim, nn_type=args.ACM_type
            )
        elif ACM_features == "AcE":
            self.classification_head = ClassificationHead(
                input_size=args.AcE_feature_size, nn_type=args.ACM_type
            )
        elif ACM_features == "combo":
            self.classification_head = ClassificationHead(
                input_size=args.AcE_feature_size + self.image_features_dim,
                nn_type=args.ACM_type,
            )

        for param in self.parameters():
            param.data = param.data.to(torch.float32)

        if args.AcE_checkpoint:
            checkpoint = torch.load(args.AcE_checkpoint)
            self.head.load_state_dict(checkpoint)
            print(f"AcE was loded from {args.AcE_checkpoint}")

        if args.ACM_checkpoint:
            checkpoint = torch.load(args.ACM_checkpoint)
            self.classification_head.load_state_dict(checkpoint)
            print(f"ACM was loded from {args.ACM_checkpoint}")

    def encode_mae(self, images):
        image_features = self.image_encoder.forward_encoder(images, 0)
        image_features = self.image_encoder.norm(image_features[0].mean(1))
        return image_features

    def forward(self, images):
        features = self.encode_image(images)
        features = self.head(features)
        return features

    def forward_image_features(self, features):
        if self.head_type == "MLP":
            features = features.to(torch.float32)
            AcE_features = self.head(features)
        else:
            features = features.to(torch.float32)
            features = features.unsqueeze(0)
            AcE_features = self.head(features).squeeze(0)
        return AcE_features

    def predict_aff_image_features(self, features):
        if self.ACM_features == "image":
            if self.args.ACM_type == "Hopfield":
                predictions = self.classification_head(
                    features.detach().float().unsqueeze(0)
                ).squeeze(0)
            else:
                predictions = self.classification_head(features.detach().float())
        elif self.ACM_features == "AcE":
            if self.args.ACM_type == "Hopfield":
                features = self.forward_image_features(features).detach().unsqueeze(0)
                predictions = self.classification_head(features).squeeze(0)
            else:
                features = self.forward_image_features(features).detach()
                predictions = self.classification_head(features)
        elif self.ACM_features == "combo":
            if self.args.ACM_type == "Hopfield":
                AcE_features = self.forward_image_features(features).detach()
                features = torch.cat((AcE_features, features), dim=1).unsqueeze(0)
                predictions = self.classification_head(features).squeeze(0)
            else:
                AcE_features = self.forward_image_features(features).detach()
                features = torch.cat((AcE_features, features), dim=1)
                predictions = self.classification_head(features)
        return predictions
