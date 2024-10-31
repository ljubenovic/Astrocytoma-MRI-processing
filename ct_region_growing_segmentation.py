import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from skimage import measure
from utils import *
from visualisation import *


def ctrg_segmentation(mri_npy, idx_start, idx_end, seed, thr_low, thr_high, idx_focus):

    print('data type: ', mri_npy.dtype)
    print('min val: ', np.min(mri_npy), '\nmax val: ', np.max(mri_npy))

    plt.figure()
    plt.imshow(mri_npy[idx,:,:], cmap='gray')
    plt.scatter(seed[1], seed[0], c='red', label='Seed')
    plt.title('TU {}'.format(astrocytoma_grade))
    plt.legend()
    plt.show()

    mri_sitk = sitk.GetImageFromArray(mri_npy, isVector=False)
    
    seed = [(seed[0],seed[1],idx) for idx in range(idx_start,idx_end+1)]

    seg = sitk.Image(mri_sitk.GetSize(), sitk.sitkUInt8)
    seg.CopyInformation(mri_sitk)

    for s in seed:
        seg[s] = 1  

    seg = sitk.BinaryDilate(seg, [3]*3)

    seed_overlay=sitk.LabelOverlay(mri_sitk, seg, opacity=0.5, backgroundValue=0)
    seed_overlay=sitk.GetArrayFromImage(seed_overlay)
    seed_overlay=seed_overlay[:,:,:,0]

    seg_CTRG = sitk.ConnectedThreshold(mri_sitk, seedList=seed, lower=thr_low, upper=thr_high)

    """
    # Izbacivanje iz maske onih piksela koji nisu direktno povezani sa seed-om

    mask_sitk = seg_CTRG
    connected_components = sitk.ConnectedComponent(mask_sitk)

    # Pronađi labelu regiona koji sadrži seed
    label_shape_statistics = sitk.LabelShapeStatisticsImageFilter()
    label_shape_statistics.Execute(connected_components)
    
    seed = seed[idx_focus-idx_start]
    seed = (seed[0], seed[1], idx_focus)
    print(seed)
    seed_label = connected_components[seed]

    filtered_mask = sitk.Equal(connected_components, seed_label)

    seg_CTRG = filtered_mask
    """
    
    """
    # Popunjavanje rupica

    dilated = sitk.BinaryDilate(seg_CTRG, [2]*seg_CTRG.GetDimension())
    filled_mask = sitk.BinaryErode(dilated, [2]*seg_CTRG.GetDimension())
    
    seg_CTRG = filled_mask
    """

    mask = sitk.GetArrayFromImage(seg_CTRG)

    return mask


if __name__ == '__main__':

    astrocytoma_grade = 4
    plane = 'S'

    if astrocytoma_grade == 2:
        idx = 50
        idx_start = 46
        idx_end = 55

        thr_low = 19
        thr_high = 24

        seed = (89,110)

    elif astrocytoma_grade == 3:
        pass

    else:
        idx = 43
        idx_start = 36
        idx_end = 53

        thr_low = 17
        thr_high = 40

        seed = (77,132)       

    # Ucitavanje i prikaz MRI snimka
    mri_npy = load_dcm_slices(astrocytoma_grade, plane)
    plot_sliding_slices(mri_npy, idx_start, idx_end)

    # Segmentacija tumora primenom CTRG i prikaz rezultata
    mask = ctrg_segmentation(mri_npy, idx_start, idx_end, seed, thr_low, thr_high, idx)
    mri_masked = mri_npy*mask
    plot_sliding_slices(mri_masked, idx_start, idx_end)

    # Poredjenje rezultata segmentacije sa rucno segmentisanim tumorom
    mask_npy = load_mask_stack(astrocytoma_grade, plane, 'manual')

    mri_slice = mri_npy[idx,:,:]
    mask_slice = mask[idx,:,:]

    mask_slice_nan = np.where(mask_slice,1,np.nan)

    plt.figure()
    plt.imshow(mri_slice, cmap='gray')
    plt.imshow(mask_slice_nan)
    contours = measure.find_contours(mask_npy[idx,:,:])
    for contour in contours:
        plt.plot(contour[:, 1], contour[:, 0], '-r', linewidth=1)
    plt.axis('off')
    plt.show()
