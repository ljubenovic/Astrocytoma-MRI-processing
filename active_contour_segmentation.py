import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import gaussian
from skimage.measure import find_contours
from skimage.segmentation import active_contour
from skimage.draw import polygon2mask
from utils import *
from visualisation import *


def active_contour_segmentation(mri_slices, mask_slices_init, plotting = False, plot_ind = None):

    n_slices, height, width = mri_slices.shape
    blank_slice = np.zeros((height, width))

    mask = []

    for i in range(n_slices):

        print(i)
        
        mri_slice = mri_slices[i,:,:]
        mask_slice_init = mask_slices_init[i,:,:]

        if np.any(mask_slice_init):

            mri_smooth = gaussian(mri_slice, sigma=0)

            contours = find_contours(mask_slice_init, level=0.5)
            
            init_contour = contours[0]
            init_contour = np.repeat(init_contour, 20, axis=0)

            snake = active_contour(mri_smooth, init_contour, alpha=0.08, beta=1, gamma=0.004, max_num_iter=1000)

            mask_slice = polygon2mask(image_shape=(height, width), polygon=snake)
            mask.append(mask_slice)

            if plotting and i == plot_ind:
                
                    mask_slice_nan = np.where(mask_slice==0, np.nan, 1)

                    plt.figure(figsize=(16,8))
                    
                    plt.subplot(1,2,1)
                    plt.imshow(mri_slice, cmap=plt.cm.gray)
                    plt.plot(init_contour[:,1], init_contour[:,0], '--y', linewidth=0.8, label='initial contour')
                    plt.plot(snake[:, 1], snake[:, 0], 'purple', label='final contour')
                    plt.legend()
                    plt.axis('off')

                    plt.subplot(1,2,2)
                    plt.imshow(mri_slice, cmap=plt.cm.gray)
                    plt.imshow(mask_slice_nan, alpha=0.4)
                    plt.axis('off')

                    plt.show()
                    plt.savefig(os.path.join('results','active_cont_TU_{}_{}_{}.png'.format(astrocytoma_grade,plane,i)))
                    plt.close()
        else:
            mask.append(blank_slice)

    mask = np.stack(mask, axis=0)
    
    return mask


if __name__ == '__main__':

    astrocytoma_grade = 4      # 2, 3, 4
    plane = "S"                # C, S, T

    plot_ind = 43
    idx_start = 36
    idx_end = 53

    mri_stack = load_dcm_slices(astrocytoma_grade, plane)
    mask_stack_init = load_mask_stack(astrocytoma_grade, plane, 'manual')

    mask_stack = active_contour_segmentation(mri_stack, mask_stack_init, plotting = False, plot_ind = plot_ind)

    mask_path = os.path.join("active contour labels", "TU " + str(astrocytoma_grade), plane, "mask.npy")
    os.makedirs(os.path.dirname(mask_path), exist_ok=True)
    np.save(mask_path, mask_stack)

    masked_mri = mask_stack*mri_stack
    plot_sliding_slices(masked_mri, idx_start, idx_end)

    