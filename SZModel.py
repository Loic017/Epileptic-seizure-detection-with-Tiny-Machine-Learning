import torch
import torch.nn as nn
from torchvision.models.feature_extraction import create_feature_extractor
import torch.nn.functional as F


class SZModel(nn.Module):
    def __init__(self, l1=128):
        super(SZModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, (3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(
            16, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
        )
        self.hardswish = nn.Hardswish()
        self.conv1_2 = nn.Conv2d(
            16, 16, (3, 3), stride=(2, 2), padding=(1, 1), bias=False
        )
        self.ReLU = nn.ReLU()

        self.block1 = nn.Sequential(
            nn.Conv2d(
                16, 16, (3, 3), stride=(1, 1), padding=(1, 1), groups=16, bias=False
            ),
            nn.BatchNorm2d(
                16, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
            ),
            nn.ReLU(),
            nn.Conv2d(16, 16, (1, 1), stride=(1, 1), padding=(0, 0), bias=False),
            nn.BatchNorm2d(
                16, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
            ),
        )

        self.conv2 = nn.Conv2d(
            16, 16, (3, 3), stride=(2, 2), padding=(1, 1), bias=False
        )
        self.bn2 = nn.BatchNorm2d(
            16, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
        )
        self.hardswish = nn.Hardswish()

        self.block2 = nn.Sequential(
            nn.Conv2d(
                16, 16, (3, 3), stride=(1, 1), padding=(1, 1), groups=16, bias=False
            ),
            nn.BatchNorm2d(
                16, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
            ),
            nn.ReLU(),
            nn.Conv2d(16, 16, (1, 1), stride=(1, 1), padding=(0, 0), bias=False),
            nn.BatchNorm2d(
                16, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
            ),
        )

        self.globalaveragepool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.classif_block = nn.Sequential(
            nn.Linear(16, l1),
            nn.Hardswish(),
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(l1, 2),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.hardswish(x)
        x = self.conv1_2(x)
        x = self.ReLU(x)

        x = self.block1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.hardswish(x)

        x = self.block2(x)

        x = self.globalaveragepool(x)
        x = self.flatten(x)

        x = self.classif_block(x)
        return x


def combined_models_load(device):
    model_absence = SZModel()
    model_absence.load_state_dict(
        torch.load("model/pytorch_models/default/absence_model.pt")
    )

    model_absence = model_absence.to(device)
    return_nodes = {
        "block2.4": "rr",
    }
    model_absence = create_feature_extractor(model_absence, return_nodes=return_nodes)

    model_tonic_clonic = SZModel()
    model_tonic_clonic.load_state_dict(
        torch.load("model/pytorch_models/default/tonic-clonic_model.pt")
    )
    model_tonic_clonic = model_tonic_clonic.to(device)
    return_nodes = {
        "block2.4": "rr",
    }
    model_tonic_clonic = create_feature_extractor(
        model_tonic_clonic, return_nodes=return_nodes
    )

    model_general = SZModel()
    model_general.load_state_dict(
        torch.load("model/pytorch_models/default/general_model.pt")
    )
    model_general = model_general.to(device)
    return_nodes = {
        "block2.4": "rr",
    }
    model_general = create_feature_extractor(model_general, return_nodes=return_nodes)

    return model_absence, model_tonic_clonic, model_general


class SZModel_Combined(nn.Module):
    def __init__(self, absence_model, tonic_clonic_model, general_model):
        super(SZModel_Combined, self).__init__()
        self.model_absence = absence_model
        self.model_tonic_clonic = tonic_clonic_model
        self.model_general = general_model
        self.conv1 = nn.Conv2d(
            48, 16, (1, 1), stride=(1, 1), padding=(0, 0), bias=False
        )
        self.globalaveragepool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.classif_block = nn.Sequential(
            nn.Linear(16, 32),
            nn.Hardswish(),
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(32, 4),
        )

    def forward(self, x):
        a = self.model_absence(x)
        b = self.model_tonic_clonic(x)
        c = self.model_general(x)

        a, b, c = a["rr"], b["rr"], c["rr"]
        x = torch.cat((a, b, c), dim=1)
        x = F.relu(self.conv1(x))
        x = self.globalaveragepool(x)
        x = self.flatten(x)
        x = self.classif_block(x)
        return x


class SZModel_Softmax(nn.Module):
    def __init__(self, l1=128):
        super(SZModel_Softmax, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, (3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(
            16, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
        )
        self.hardswish = nn.Hardswish()
        self.conv1_2 = nn.Conv2d(
            16, 16, (3, 3), stride=(2, 2), padding=(1, 1), bias=False
        )
        self.ReLU = nn.ReLU()

        self.block1 = nn.Sequential(
            nn.Conv2d(
                16, 16, (3, 3), stride=(1, 1), padding=(1, 1), groups=16, bias=False
            ),
            nn.BatchNorm2d(
                16, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
            ),
            nn.ReLU(),
            nn.Conv2d(16, 16, (1, 1), stride=(1, 1), padding=(0, 0), bias=False),
            nn.BatchNorm2d(
                16, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
            ),
        )

        self.conv2 = nn.Conv2d(
            16, 16, (3, 3), stride=(2, 2), padding=(1, 1), bias=False
        )
        self.bn2 = nn.BatchNorm2d(
            16, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
        )
        self.hardswish = nn.Hardswish()

        self.block2 = nn.Sequential(
            nn.Conv2d(
                16, 16, (3, 3), stride=(1, 1), padding=(1, 1), groups=16, bias=False
            ),
            nn.BatchNorm2d(
                16, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
            ),
            nn.ReLU(),
            nn.Conv2d(16, 16, (1, 1), stride=(1, 1), padding=(0, 0), bias=False),
            nn.BatchNorm2d(
                16, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
            ),
        )

        self.globalaveragepool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.classif_block = nn.Sequential(
            nn.Linear(16, l1),
            nn.Hardswish(),
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(l1, 2),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.hardswish(x)
        x = self.conv1_2(x)
        x = self.ReLU(x)

        x = self.block1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.hardswish(x)

        x = self.block2(x)

        x = self.globalaveragepool(x)
        x = self.flatten(x)

        x = self.classif_block(x)
        return x


class SZModel_SIGMOID(nn.Module):
    def __init__(self, l1=128):
        super(SZModel_SIGMOID, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, (3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(
            16, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
        )
        self.hardswish = nn.Hardswish()
        self.conv1_2 = nn.Conv2d(
            16, 16, (3, 3), stride=(2, 2), padding=(1, 1), bias=False
        )
        self.ReLU = nn.ReLU()

        self.block1 = nn.Sequential(
            nn.Conv2d(
                16, 16, (3, 3), stride=(1, 1), padding=(1, 1), groups=16, bias=False
            ),
            nn.BatchNorm2d(
                16, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
            ),
            nn.ReLU(),
            nn.Conv2d(16, 16, (1, 1), stride=(1, 1), padding=(0, 0), bias=False),
            nn.BatchNorm2d(
                16, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
            ),
        )

        self.conv2 = nn.Conv2d(
            16, 16, (3, 3), stride=(2, 2), padding=(1, 1), bias=False
        )
        self.bn2 = nn.BatchNorm2d(
            16, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
        )
        self.hardswish = nn.Hardswish()

        self.block2 = nn.Sequential(
            nn.Conv2d(
                16, 16, (3, 3), stride=(1, 1), padding=(1, 1), groups=16, bias=False
            ),
            nn.BatchNorm2d(
                16, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
            ),
            nn.ReLU(),
            nn.Conv2d(16, 16, (1, 1), stride=(1, 1), padding=(0, 0), bias=False),
            nn.BatchNorm2d(
                16, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
            ),
        )

        self.globalaveragepool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.classif_block = nn.Sequential(
            nn.Linear(16, l1),
            nn.Hardswish(),
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(l1, 2),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.hardswish(x)
        x = self.conv1_2(x)
        x = self.ReLU(x)

        x = self.block1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.hardswish(x)

        x = self.block2(x)

        x = self.globalaveragepool(x)
        x = self.flatten(x)

        x = self.classif_block(x)
        return x
