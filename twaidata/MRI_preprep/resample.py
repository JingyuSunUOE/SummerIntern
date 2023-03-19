import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import os


# resamples an image
def get_resampled_img(itk_image, out_spacing=[2.0, 2.0, 2.0], is_label=False):
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()

    # what's this? its how to modify the output size I think...
    out_size = [
        int(np.round(orig_size * orig_spacing / out_spacing))
        for (orig_size, orig_spacing, out_spacing)
        in zip(original_size, original_spacing, out_spacing)
    ]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())  # sets the output direction cosine matrix...
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)

    return resample.Execute(itk_image)


def save_file(filepath, image):
    writer = sitk.ImageFileWriter()
    writer.SetFileName(filepath)
    writer.Execute(image)


# io wrap on resample fuc function
def resample_and_save(filename_in, filename_out, is_label=False, outspacing=[1., 1., 3.], overwrite=False):
    """
    resamples an image to custom voxel dimensions and saves result to disk
    
    filename_in: .nii.gz file to load
    filename_out: save location
    is_label: is the file a brain scan or a segmentaion mask
    outspacing: the spacing of the image to be resampled to.
    overwrite: if the file already exists in the out
    dir, replace it with a newly resampled image if True else no do nothing and return.
    """

    # check that the file doesn't already exist in the target domain
    if os.path.exists(filename_out):
        if not overwrite:
            return

    # do resample and save
    image = sitk.ReadImage(filename_in)
    resampled_image = get_resampled_img(image, outspacing, is_label)
    save_file(filename_out, resampled_image)


def resample_and_return(filename_in, is_label=False, out_spacing=[1., 1., 3.]):
    """
    resamples an image to custom voxel dimensions and returns image as a numpy array
    
    filename_in: .nii.gz file to load
    is_label: is the file a brain scan or a segmentaion mask
    outspacing: the spacing of the image to be resampled to.
    """

    # resample the image
    image = sitk.ReadImage(filename_in)
    resampled_image = get_resampled_img(image, out_spacing, is_label)

    return sitk.GetArrayFromImage(resampled_image)
