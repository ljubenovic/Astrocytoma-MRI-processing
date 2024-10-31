from skimage import feature
from utils import *
from visualisation import *


def blood_vessel_segmentation(mri_slices, mask_slices, sigma=0):

    (n_slices, _, _) = mri_slices.shape

    vessels_mask = np.zeros_like(mask_slices)

    for i in range(n_slices):

        mri_slice = mri_slices[i,:,:]
        mask_slice = mask_slices[i,:,:]

        edges = feature.canny(mri_slice, sigma=sigma)
        edges_masked = edges*mask_slice

        vessels_mask[i,:,:] = edges_masked

    mri_masked = mri_slices*mask_slices

    mri_vessels = np.where(vessels_mask, 255, mri_masked)

    return (vessels_mask, mri_vessels)



if __name__ == '__main__':

    astrocytoma_grade = 4
    plane = 'S'

    mri_stack = load_dcm_slices(astrocytoma_grade, plane)
    mask_stack = load_mask_stack(astrocytoma_grade, plane,'manual')

    (vessels_mask, mri_vessels) = blood_vessel_segmentation(mri_stack, mask_stack, sigma=0)

    volume_rendering(mri_vessels, background=(0.5,0.5,0.5))
    volume_rendering(mri_vessels, background=(0,0,0))

