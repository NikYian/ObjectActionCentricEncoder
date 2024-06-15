import torch
import torch.nn as nn
import clip
from models.teacher import load_teacher
from torch.nn.functional import softmax

import torch.nn as nn


class AcEnn(nn.Module):
    def __init__(self, args):
        super(AcEnn, self).__init__()
        self.args = args

        self.clip, self.preprocess = clip.load(args.CLIP_model, device=args.device)
        for param in self.clip.parameters():  # CLIP params are frozen
            param.requires_grad = False
        # for param in self.preprocess.parameters():
        #     param.requires_grad = False

        # self.head = nn.Linear(self.clip.visual.output_dim, args.AcE_feature_size)

        # self.head = nn.Sequential(
        #     nn.Linear(self.clip.visual.output_dim, args.AcE_hidden_layers[0]),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(args.AcE_dropout_rate),
        #     nn.Linear(args.AcE_hidden_layers[0], args.AcE_hidden_layers[1]),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(args.AcE_dropout_rate),
        #     nn.Linear(args.AcE_hidden_layers[1], args.AcE_hidden_layers[2]),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(args.AcE_dropout_rate),
        #     nn.Linear(args.AcE_hidden_layers[2], args.AcE_hidden_layers[3]),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(args.AcE_dropout_rate),
        #     nn.Linear(args.AcE_hidden_layers[3], args.AcE_feature_size),
        # )

        layers = []
        layers.append(nn.Linear(self.clip.visual.output_dim, args.AcE_hidden_layers[0]))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Dropout(args.AcE_dropout_rate))

        # Add hidden layers
        for i in range(len(args.AcE_hidden_layers) - 1):
            layers.append(
                nn.Linear(args.AcE_hidden_layers[i], args.AcE_hidden_layers[i + 1])
            )
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(args.AcE_dropout_rate))

        # Add output layer
        layers.append(nn.Linear(args.AcE_hidden_layers[-1], args.AcE_feature_size))
        self.head = nn.Sequential(*layers)
        self.classification_head = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(args.AcE_feature_size, 10),
            nn.Sigmoid(),
        )

        self.relu = torch.nn.ReLU()
        self.aff_anchors = None
        self.aff_anchors_CLIP = None

        self.temperatures = nn.Parameter(torch.ones(10) * args.temperature_init)
        # self.ac_head = load_teacher(args).head  # action classification head

        for param in self.parameters():
            param.data = param.data.to(torch.float32)

        if args.AcE_checkpoint:
            checkpoint = torch.load(args.AcE_checkpoint)
            self.head.load_state_dict(checkpoint)

    def forward(self, images):
        # images = self.preprocess(images)
        clip_features = self.clip.encode_image(images)
        features = self.head(clip_features)
        return features

    def forward_text(self, texts):
        tokenized_text = clip.tokenize(texts)
        tokenized_text = tokenized_text.to(self.args.device)
        clip_features = self.clip.encode_text(tokenized_text)
        features = self.head(clip_features)
        return features, clip_features

    def forward_CLIP(self, features):
        features = features.to(torch.float32)
        features = self.head(features)
        return features

    def forward_CLIP_action_pred(self, features):
        features = features.to(torch.float32)
        features = self.head(features)
        similarities = self.ZS_predict(features, cosine=True)
        return features, similarities

    def update_aff_anchors(self):
        self.aff_anchors = []
        self.aff_anchors_CLIP = []
        for aff_sentense in self.args.affordance_sentences:
            anchor_features, clip_features = self.forward_text(aff_sentense)
            anchor_features.detach()
            clip_features.detach()
            self.aff_anchors.append(anchor_features)
            self.aff_anchors_CLIP.append(clip_features)
        self.aff_anchors = torch.cat(self.aff_anchors, dim=0)
        self.aff_anchors_CLIP = torch.cat(self.aff_anchors_CLIP, dim=0)
        # print(self.aff_anchors[0][:10])

    def ZS_predict(self, AcE_features, cosine=True):

        if cosine:
            features_norm = torch.nn.functional.normalize(AcE_features, dim=1)
            anchors_norm = torch.nn.functional.normalize(self.aff_anchors, dim=1)
            similarities = torch.mm(features_norm, torch.transpose(anchors_norm, 0, 1))
            similarities = (similarities + 1) / 2
            similarities = (similarities - 0.5) / 0.5
            similarities = self.relu(similarities)
            return similarities
        else:
            similarities = torch.mm(
                AcE_features, torch.transpose(self.aff_anchors, 0, 1)
            ).detach()
            temperatures = self.temperatures.view(1, -1)
            scaled_similarities = similarities * temperatures
            result = torch.sigmoid(scaled_similarities)
            return result

    def ZS_predict_CLIP(self, clip_features):
        features_norm = torch.nn.functional.normalize(clip_features, dim=1).half()
        anchors_norm = torch.nn.functional.normalize(
            self.aff_anchors_CLIP, dim=1
        ).half()
        similarities = torch.mm(features_norm, torch.transpose(anchors_norm, 0, 1))
        similarities = (similarities + 1) / 2
        # similarities = (similarities - 0.5) / 0.5
        # similarities = self.relu(similarities)
        return similarities

    def predict(self, AcE_features):
        return self.classification_head(AcE_features)
