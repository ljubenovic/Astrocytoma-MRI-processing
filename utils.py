import pydicom
import numpy as np
import os


def load_dcm_slices(astrocytoma_grade, plane):

    path = os.path.join("MRI data", "TU " + str(astrocytoma_grade), plane)
    dcm_files = [pydicom.dcmread(os.path.join(path, file)) for file in os.listdir(path) if file.endswith('.dcm')]
    if len(dcm_files[0].pixel_array.shape) == 2:
        mri_stack = np.array([slice.pixel_array for slice in dcm_files[:-1]])
    else:
        mri_stack = np.squeeze(np.array([slice.pixel_array for slice in dcm_files]))

    npy_path = os.path.join("MRI data", "TU " + str(astrocytoma_grade), plane,"data.npy")
    np.save(npy_path, mri_stack)

    mri_stack = (mri_stack / 1753 * 255).astype(np.uint8)

    if astrocytoma_grade == 4:
        # MRI snimci za TU 2 i TU 3 imaju po 71 slajs, dok za TU 4 imaju 192 slajsa
        mri_stack = mri_stack[25:-26:2,:,:]

    return mri_stack


def load_mask_stack(astrocytoma_grade, plane, type='manual'):
    """
    type = {'manual','active contour'}
    """
    mask_path = os.path.join("{} labels".format(type), "TU " + str(astrocytoma_grade), plane, "mask.npy")
    mask_stack = np.load(mask_path)

    return mask_stack


def MRI_segmentation(mri_stack, mask_stack):

    masked_mri = mri_stack*mask_stack
    #segmented_mri = np.array([slice for slice in masked_mri if np.any(slice)])

    return masked_mri


 
if __name__ == "__main__":

    astrocytoma_grade = 4   # 2, 3, 4
    plane = "S"             # C, S, T

    mri_stack = load_dcm_slices(astrocytoma_grade, plane)
    
    mask_stack = load_mask_stack(astrocytoma_grade, plane)

    mri_segmented = MRI_segmentation(mri_stack, mask_stack)

