import numpy as np
import pytorch_lightning as pl
import torch
from torchvision.utils import save_image
from PIL import Image
import cv2
from torch.utils.data import random_split, ConcatDataset, DataLoader
from trustworthai.utils.augmentation.standard_transforms import PairedCentreCrop
from trustworthai.models.base_models.torchUNet import UNet
from trustworthai.models.base_models.torchUNet import UNet3D
from trustworthai.utils.augmentation.standard_transforms import LabelSelect, RandomFlip, PairedCompose, CropZDim
from trustworthai.utils.fitting_and_inference.fitters.basic_lightning_fitter import StandardLitModelWrapper
from trustworthai.utils.losses_and_metrics.dice_losses import GeneralizedDiceLoss
from twaidata.torchdatasets.whole_brain_dataset import MRISegmentationDatasetFromFile
from twaidata.torchdatasets.in_ram_ds import MRISegmentation3DDataset, MRISegmentation2DDataset
from pathlib import Path
from twaidata.MRI_preprep.io import load_nii_img, save_nii_img, FORMAT
from twaidata.MRI_preprep.normalize_brain import normalize_brain
from twaidata.MRI_preprep.resample import resample_and_save
from twaidata.mri_dataset_directory_parsers.parser_selector import select_parser
from natsort import natsorted
import subprocess
import shutil
import os


def get_transforms_stage2(crop_size, label_extract):
    if label_extract == None:
        print("keeping all labels")
        return PairedCentreCrop(crop_size)
    else:
        print(f"extracting label {label_extract}")
        transforms = PairedCompose([
            PairedCentreCrop(crop_size),  # cut out the centre square
            LabelSelect(label_extract),  # extract the desired label
        ])
        return transforms


def preprocess_file_collation(in_dir, out_dir, domain):
    # extract args
    name = 'WMH_challenge_dataset'
    crop_height = 224
    crop_width = 160
    label_extract = None

    # check file paths are okay
    in_dir = os.path.join(in_dir, name)
    out_dir = os.path.join(out_dir, name)
    if domain != None:
        in_dir = os.path.join(in_dir, domain)
        out_dir = os.path.join(out_dir, domain)

    if not os.path.exists(in_dir):
        raise ValueError(f"could not find folder: {in_dir}")

    print(f"processing dataset: {in_dir}")

    if not os.path.exists(out_dir):
        try:
            Path(out_dir).mkdir(parents=True, exist_ok=True)
        except FileNotFoundError:
            print(f"Warning: couldn't make output directory here: {out_dir}")

    # select centre crop and optionaly label extract transform
    crop_size = (crop_height, crop_width)
    transforms = get_transforms_stage2(crop_size, label_extract)

    # load the dataset
    dataset = MRISegmentationDatasetFromFile(
        in_dir,
        img_filetypes=["FLAIR_BET_mask.nii.gz", "FLAIR.nii.gz", "T1.nii.gz"],  # brain mask, flair, T1.
        label_filetype="wmh.nii.gz",
        transforms=transforms
    )

    # collect the images and labels in to a list
    data_imgs = []
    data_labels = []
    slices = []  # check for inconsistent slice sizes across a domain
    for (img, label) in dataset:
        data_imgs.append(img)
        data_labels.append(label)
        slices.append(img.shape[1])

    # where there is more than one slice size in the domain
    # take a centre crop of the sizes equal to the miniumum
    # number of slices found in the domain.
    # should not affect the WMH challenge data, only the ED inhouse data.
    slices = np.array(slices)
    uniques = np.unique(slices)
    if len(uniques) > 1:
        print(f"unique slice sizes found in domain: {uniques}")
        # for each image select the centre minimum slice
        centre_cut = np.min(slices)
        for i in range(len(data_imgs)):
            if centre_cut < data_imgs[i].shape[1]:  # crop images larger than the biggest slice size.
                start = (data_imgs[i].shape[1] - centre_cut) // 2
                data_imgs[i] = data_imgs[i][:, start:start + centre_cut, :, :]
                data_labels[i] = data_labels[i][:, start:start + centre_cut, :, :]

    # convert to numpy arrays
    data_imgs = np.stack(data_imgs, axis=0)
    data_labels = np.stack(data_labels, axis=0)
    print(f"dataset imgs shape: {data_imgs.shape}")
    print(f"dataset labels shape: {data_labels.shape}")

    # save the files
    out_file_imgs = os.path.join(out_dir, "imgs.npy")
    out_file_labels = os.path.join(out_dir, "labels.npy")
    np.save(out_file_imgs, data_imgs)
    np.save(out_file_labels, data_labels)


def simple_preprocess_st1(in_dir, out_dir, name, start, end):
    # ======================================================================================
    # SETUP PREPROCESSING PIPELINE
    # ======================================================================================

    # get the parser that maps inputs to outputs
    # csv file used for custom datasets
    out_spacing = "1., 1., 3."
    force_replace = "False"
    parser = select_parser(name, in_dir, out_dir, None)

    # get the files to be processed
    files_map = parser.get_dataset_inout_map()
    keys = natsorted(list(files_map.keys()))

    # select range of files to preprocess
    if end == -1:
        keys = keys[start:]
    else:
        keys = keys[start:end]

    print(f"starting at individual {start} and ending at individual {end}")

    # get the fsl directory used for brain extraction and bias field correction
    FSLDIR = os.getenv('FSLDIR')
    if 'FSLDIR' == "":
        raise ValueError("FSL is not installed. Install FSL to complete brain extraction")

    # parse the outspacing argument
    outspacing = [float(x) for x in out_spacing.split(",")]
    if len(outspacing) != 3:  # 3D
        raise ValueError(f"malformed outspacing parameter: {out_spacing}")
    else:
        print(f"using out_spacing: {outspacing}")

    # ======================================================================================
    # RUN
    # ======================================================================================
    for ind in keys:
        print(f"processing individual: {ind}")
        ind_filemap = files_map[ind]

        # check whether all individuals have been done and can therefore be skipped
        can_skip = True
        if not force_replace.lower() == "true":
            for filetype in files_map[ind].keys():
                output_dir = ind_filemap[filetype]['outpath']
                output_filename = ind_filemap[filetype]['outfilename']

                if output_filename != None and not os.path.exists(os.path.join(output_dir, output_filename + FORMAT)):
                    can_skip = False
                    break
        else:
            can_skip = False

        if can_skip:
            print(f"skipping, because preprocessed individual {ind} file exists and force_replace set to false")
            continue

        for filetype in natsorted(files_map[ind].keys()):
            if filetype == "ICV":
                continue

            print(f"processing filetype: ", filetype)

            infile = ind_filemap[filetype]['infile']
            output_dir = ind_filemap[filetype]['outpath']
            output_filename = ind_filemap[filetype]['outfilename']
            islabel = ind_filemap[filetype]['islabel']

            print(f"processing file: {infile}")

            # check that the file exists
            if not os.path.exists(infile):
                raise ValueError(f"target file doesn't exist: {keys}")
            # create the output directory if it does not exist
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)

            next_file = infile.split(FORMAT)[0]
            # ======================================================================================
            # BIAS FIELD CORRECTION (T1 ONLY HERE)
            # ======================================================================================

            # only applied to T1 (the ED data doesn't need it and I'm not sure about WMH challenge...)
            if filetype == "T1":
                # define name of file to be saved
                out_file = os.path.join(output_dir, f"{output_filename}_BIAS_CORR")

                # fast outputs many files to the original folder the data is located in.
                # The relvant file (_restore.nii.gz) is copied over.
                bias_field_corr_command = [os.path.join(*[FSLDIR, 'bin', 'fast']), '-b', '-B', next_file + FORMAT]
                _ = subprocess.call(bias_field_corr_command)

                corrected_file = next_file.split(".nii.gz")[0] + "_restore.nii.gz"
                _ = subprocess.call(["cp", corrected_file, out_file + FORMAT])

                next_file = out_file
                print("outfile post bfc: ", out_file)

            # ======================================================================================
            # BRAIN EXTRACTION
            # ======================================================================================
            if not islabel:
                out_file = os.path.join(output_dir, f"{output_filename}_BET")
                flair_outdir = ind_filemap["FLAIR"]["outpath"]
                flair_outfilename = ind_filemap["FLAIR"]["outfilename"]
                mask_out_file = os.path.join(flair_outdir, flair_outfilename + "_BET_mask")

                # check to see if the file has a ICV volume file defined
                # if so, use it as the brain extraction mask, otherwise, run BET
                if "ICV" in ind_filemap and os.path.exists(ind_filemap["ICV"]["infile"]):
                    # multiply ICV mask by brain mask
                    icv_filepath = ind_filemap["ICV"]["infile"]
                    icv, _ = load_nii_img(icv_filepath)
                    img, header = load_nii_img(next_file)
                    img = img.squeeze()

                    img = img * icv.squeeze()
                    save_nii_img(out_file, img, header)

                    # copy the ICV to the output directory
                    if filetype == "FLAIR":
                        cp_command = ['cp', icv_filepath, mask_out_file + FORMAT]
                        _ = subprocess.call(cp_command)

                elif filetype != "FLAIR":
                    # if it isn't a flair file, use the processed BET flair file as a map
                    # load the BET processed flair file (flairs must be processced first):
                    bet_flair, _ = load_nii_img(mask_out_file)

                    # load the target image (say T1)
                    img, header = load_nii_img(next_file)
                    img = img.squeeze()

                    # apply the mask
                    img = img * bet_flair
                    save_nii_img(out_file, img, header)

                else:
                    # if image type is flair, generate the bet mask
                    # flair images must be run first so that the flair bet mask exists when preprocessing the t1
                    # run BET tool
                    # bet outputs the result and the mask

                    # if there should have been an icv file use bet with -S for a better extraction, otherwise just use the simpler preprocessed
                    # version for now (ideally every file will be preprocessed with the -S flag but it is a lot slower)
                    if "ICV" in ind_filemap:  # i.e others in the datset have ICV e.g for the Ed Data but for this individual the ICV file couldn't be found.
                        bet_command = [os.path.join(*[FSLDIR, 'bin', 'bet']), next_file + FORMAT, out_file, "-m", "-S"]
                    else:
                        bet_command = [os.path.join(*[FSLDIR, 'bin', 'bet2']), next_file + FORMAT, out_file, "-m"]

                    _ = subprocess.call(bet_command)

                next_file = out_file
                print("outfile post brain extract: ", out_file)

            # ======================================================================================
            # NORMALIZE
            # ======================================================================================
            if not islabel:
                # do the normalizing
                img, header = load_nii_img(next_file)
                img = img.squeeze()
                normalize_brain(img)  # in place operation

                # save the results
                out_file = os.path.join(output_dir, f"{output_filename}_NORMALIZE")
                save_nii_img(out_file, img, header)

                next_file = out_file
                print("outfile post normalize: ", out_file)

            # ======================================================================================
            # RESAMPLE
            # ======================================================================================
            out_file = os.path.join(output_dir, output_filename)  # last step in preprocessing order
            resample_and_save(next_file, out_file + FORMAT, is_label=islabel, outspacing=outspacing)
            print("outfile post resample: ", out_file)

        # resample the brain mask
        flair_outdir = ind_filemap["FLAIR"]["outpath"]
        flair_outfilename = ind_filemap["FLAIR"]["outfilename"]
        mask_file = os.path.join(flair_outdir, flair_outfilename + "_BET_mask" + FORMAT)
        resample_and_save(mask_file, mask_file, is_label=True, outspacing=outspacing, overwrite=True)
        print()


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


# augmentation definition
def get_transforms(is3D):
    transforms = [
        LabelSelect(label_id=1),
        RandomFlip(p=0.5, orientation="horizontal"),
    ]
    if not is3D:
        return PairedCompose(transforms)
    else:
        transforms.append(CropZDim(size=32, minimum=0, maximum=-1))
        return PairedCompose(transforms)


def save_images(val_dataloader, x_new, y1_hat):
    count = 0
    for i in range(32):
        x, y = next(iter(val_dataloader))
        x1 = x[0].unsqueeze(0)
        num = str(count)
        num2 = str(count + 1)
        num3 = str(count + 2)
        path = 'input/' + 'img' + num + '.png'
        path2 = 'input/' + 'img' + num2 + '.png'
        path3 = 'input/' + 'img' + num3 + '.png'
        # path4 = 'tiff/' + 'img' + num3 + '.tiff'
        img1 = x_new[0][0][i, :, :]
        save_image(img1, path)
        img2 = y1_hat[0][0][i, :, :]
        save_image(img2, path2)

        img1 = Image.open(path)
        img1 = img1.convert("RGBA")
        img2 = Image.open(path2)

        # 将图片分成小方块
        # Divide the image into small squares
        img_array = img2.load()
        # 遍历每一个像素块，并处理颜色
        # Traverse each pixel block and process the color
        width, height = img2.size  # 获取宽度和高度 get width and height
        for x in range(0, width):
            for y in range(0, height):
                rgb = img_array[x, y]  # 获取一个像素块的rgb Get the rgb of a pixel block
                r = rgb[0]
                g = rgb[1]
                b = rgb[2]
                if b > 220 and r > 220 and g > 220:  # 判断规则 judgment rule
                    img_array[x, y] = (255, 0, 0)
        img2.save(path2)

        # Import the image
        file_name = path2
        # Read the image
        src = cv2.imread(file_name, 1)
        # Convert image to image gray
        tmp = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        # Applying thresholding technique
        _, alpha = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY)
        # Using cv2.split() to split channels
        # of coloured image
        b, g, r = cv2.split(src)
        # Making list of Red, Green, Blue
        # Channels and alpha
        rgba = [b, g, r, alpha]
        # Using cv2.merge() to merge rgba
        # into a coloured/multi-channeled image
        dst = cv2.merge(rgba, 4)
        # Writing and saving to a new image
        cv2.imwrite(path2, dst)

        img2 = Image.open(path2)
        img2 = img2.convert("RGBA")
        img1.paste(img2, (0, 0), mask=img2)
        img1.save(path3)
        # img1.save(path4,'TIFF')
        count = count + 3


def main(root_path):
    seed = 3407
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    root_dir = root_path
    wmh_dir = "WMH_challenge_dataset/"

    domains = [
        wmh_dir + d for d in ["Singapore", "Utrecht", "GE3T"]
    ]

    # function to do train validate test split
    test_proportion = 0.1
    validation_proportion = 0.2

    is3D = True

    if is3D:
        transforms = CropZDim(size=32, minimum=0, maximum=-1)
        print(domains)
        datasets_domains = [MRISegmentation3DDataset(root_dir, domain, transforms=get_transforms(is3D)) for domain in
                            domains]
        # print(datasets_domains)
    else:
        datasets_domains = [MRISegmentation2DDataset(root_dir, domain, transforms=get_transforms(is3D)) for domain in
                            domains]

    # split into train, val test datasets
    datasets = [train_val_test_split(dataset, validation_proportion, test_proportion, seed) for dataset in
                datasets_domains]

    # concat the train val test datsets
    train_dataset = ConcatDataset([ds[0] for ds in datasets])
    val_dataset = ConcatDataset([ds[1] for ds in datasets])
    test_dataset = ConcatDataset([ds[2] for ds in datasets])

    # define dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)

    batch = next(iter(train_dataloader))

    in_channels = 3
    out_channels = 1
    # TODO: put in the path to the assets on your machine here
    checkpoint_dir = "assets/"

    if is3D:
        model = UNet3D(in_channels, out_channels, init_features=16, dropout_p=0., softmax=False)
        checkpoint = checkpoint_dir + "WMHCHAL_unet3d_16_no_softmax_dropout0.ckpt"
    else:
        model = UNet(in_channels, out_channels, init_features=32, dropout_p=0., softmax=False)
        checkpoint = checkpoint_dir + "WMHCHAL_unet2d_32_no_softmax_dropout0.ckpt"

    loss = GeneralizedDiceLoss(normalization='sigmoid')

    model = StandardLitModelWrapper.load_from_checkpoint(checkpoint, model=model, loss=loss)

    accelerator = "mps"
    devices = 1
    precision = 16

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        precision=precision,
    )

    # trainer.validate(model, val_dataloader)
    trainer.test(model, test_dataloader)

    # get some predictions
    x, y = next(iter(val_dataloader))
    x1 = x[0].unsqueeze(0)
    with torch.no_grad():
        y1_hat = model(x1)
    y1_hat = torch.sigmoid(y1_hat)
    y1_hat = y1_hat.detach().cpu()

    x_new = (x1 - x1.min()) / (x1.max() - x1.min())

    shutil.rmtree('input')
    os.mkdir('input')

    save_images(val_dataloader, x_new, y1_hat)

# if __name__ == '__main__':
#     main()
