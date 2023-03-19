print("importing")

import torch
import numpy as np

# dataset
from twaidata.torchdatasets.in_ram_ds import MRISegmentation2DDataset, MRISegmentation3DDataset
from torch.utils.data import DataLoader, random_split, ConcatDataset

# model
from trustworthai.models.base_models.source_kinet import kiunet, reskiunet, densekiunet, kiunet3d
from trustworthai.models.base_models.torchUNet import UNet, UNet3D
from trustworthai.models.base_models.deepmedic import DeepMedic

# augmentation and pretrain processing
from trustworthai.utils.augmentation.standard_transforms import RandomFlip, GaussianBlur, GaussianNoise, \
    RandomResizeCrop, RandomAffine, \
    NormalizeImg, PairedCompose, LabelSelect, \
    PairedCentreCrop, CropZDim
# loss function
from trustworthai.utils.losses_and_metrics.tversky_loss import TverskyLoss
from trustworthai.utils.losses_and_metrics.misc_metrics import IOU
from trustworthai.utils.losses_and_metrics.dice import dice, DiceMetric
from trustworthai.utils.losses_and_metrics.dice_losses import DiceLoss, GeneralizedDiceLoss
from trustworthai.utils.losses_and_metrics.power_jaccard_loss import PowerJaccardLoss
from torch.nn import BCELoss, MSELoss, BCEWithLogitsLoss

# fitter
from trustworthai.utils.fitting_and_inference.fitters.basic_lightning_fitter import StandardLitModelWrapper
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import pytorch_lightning as pl

# misc
import os

# ======================================================================================================================#
# CONSTS
# ======================================================================================================================#
seed = 3407

test_proportion = 0.1
validation_proportion = 0.2

checkpoint_dir = "/disk/scratch/s2208943/results/"
root_dir = "/disk/scratch/s2208943/ipdis/preprep/out_data/collated/"
wmh_dir = root_dir + "WMH_challenge_dataset/"
ed_dir = root_dir + "EdData/"

domains = [
    wmh_dir + d for d in ["Singapore", "Utrecht", "GE3T"]
]

is3D = False

z_crop_size = 32  # how many slices to randomly use as input to the 3D model
in_channels = 3  # brain mask, falir, t1
out_channels = 1  # wmh label
label_id = 1  # which label we are interested in learning (label 1 is WMH in WMH challenge dataset)

# ======================================================================================================================#
# SET SEED
# ======================================================================================================================#
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


# ======================================================================================================================#
# GET DATASET AND DATALOADER
# ======================================================================================================================#
# augmentation definintion
def get_transforms(is_3D):
    transforms = [
        LabelSelect(label_id=label_id),
        RandomFlip(p=0.5, orientation="horizontal"),
        # GaussianBlur(p=0.5, kernel_size=7, sigma=(.1, 1.5)),
        # GaussianNoise(p=0.2, mean=0, sigma=0.2),
        # RandomAffine(p=0.2, shear=(.1,3.)),
        # RandomAffine(p=0.2, degrees=5),
        # RandomResizeCrop(p=1., scale=(0.6, 1.), ratio=(3./4., 4./3.))
    ]
    if not is_3D:
        return PairedCompose(transforms)
    else:
        transforms.append(CropZDim(size=z_crop_size, minimum=0, maximum=-1))
        return PairedCompose(transforms)


# function to do train validate test split
def train_val_test_split(dataset, val_prop, test_prop, seed):
    # I think the sklearn version might be prefereable for determinism and things
    # but that involves fiddling with the dataset implementation I think....
    size = len(dataset)
    test_size = int(test_prop * size)
    val_size = int(val_prop * size)
    train_size = size - val_size - test_size
    train, val, test = random_split(dataset, [train_size, val_size, test_size],
                                    generator=torch.Generator().manual_seed(seed))
    return train, val, test


print("loading data")
# load datasets
# this step is quite slow, all the data is being loaded into memory
if is3D:
    datasets_domains = [MRISegmentation3DDataset(root_dir, domain, transforms=get_transforms(is_3D=True)) for domain in
                        domains]
else:
    datasets_domains = [MRISegmentation2DDataset(root_dir, domain, transforms=get_transforms(is_3D=False)) for domain in
                        domains]

# split into train, val test datasets
datasets = [train_val_test_split(dataset, validation_proportion, test_proportion, seed) for dataset in datasets_domains]

# concat the train val test datsets
train_dataset = ConcatDataset([ds[0] for ds in datasets])
val_dataset = ConcatDataset([ds[1] for ds in datasets])
test_dataset = ConcatDataset([ds[2] for ds in datasets])

# define dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)
val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)

# ======================================================================================================================#
# SETUP MODEL
# ======================================================================================================================#
print("loading model")
if is3D:
    model = UNet3D(in_channels, out_channels, init_features=16, dropout_p=0., softmax=False)
    optimizer_params = {"lr": 2e-3}
    lr_scheduler_params = {"step_size": 100, "gamma": 0.5}
else:
    model = UNet(in_channels, out_channels, init_features=32, dropout_p=0., softmax=False)
    optimizer_params = {"lr": 1e-3}
    lr_scheduler_params = {"step_size": 30, "gamma": 0.1}

loss = GeneralizedDiceLoss(normalization='sigmoid')
model = model = StandardLitModelWrapper(model, loss, optimizer_params=optimizer_params,
                                        lr_scheduler_params=lr_scheduler_params)

strategy = None
# strategy = "deepspeed_stage_2"
# strategy = "dp"
# strategy = "deepspeed_stage_2_offload"

accelerator = "gpu"
devices = 1
max_epochs = 1000
precision = 16

checkpoint_callback = ModelCheckpoint(checkpoint_dir, save_top_k=2, monitor="val_loss")
early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=100, verbose="False", mode="min",
                                    check_finite=True)
trainer = pl.Trainer(
    callbacks=[checkpoint_callback, early_stop_callback],
    accelerator=accelerator,
    devices=devices,
    max_epochs=max_epochs,
    strategy=strategy,
    precision=precision,
    default_root_dir=checkpoint_dir
)

# ======================================================================================================================#
# TRAIN
# ======================================================================================================================#
print("training")
trainer.fit(model, train_dataloader, val_dataloader)

# checkpoints are saved automatically in the checkpoint folder you pick.
