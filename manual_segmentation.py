import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path
from utils import *
from visualisation import *


def select_ROI(image):

    def onselect(verts):
        global mask_img, mask
        path = Path(verts)
        x, y = np.meshgrid(np.arange(mask_img.shape[1]), np.arange(mask_img.shape[0]))
        points = np.vstack((x.flatten(), y.flatten())).T
        mask = path.contains_points(points).reshape(mask_img.shape)
        ax.clear()
        ax.imshow(mask_img, cmap='gray')
        ax.imshow(mask, cmap='Reds', alpha=0.5)
        plt.draw()        

    def toggle_selector(event):
        if event.key in ['Q', 'q']:
            plt.close()

    global mask_img, ax, mask
    mask_img = image.copy()
    mask = np.zeros_like(image, dtype=np.uint8)

    fig, ax = plt.subplots()
    ax.imshow(image, cmap='gray')
    lasso = LassoSelector(ax, onselect=onselect)
    plt.connect('key_press_event', toggle_selector)
    plt.show()
    
    return mask


def manual_segmentation(mri_stack, slice_start, slice_end):

    (n_slices, height, width) = mri_stack.shape

    stack = mri_stack[slice_start:slice_end + 1]
    mask_stack = np.stack([select_ROI(stack[i,:,:]) for i in range(stack.shape[0])], axis=0)

    blank_slice = np.zeros((height, width), dtype=mask_stack.dtype)

    n_blank_start = slice_start - 1
    n_blank_end = n_slices - slice_end

    blank_start = np.tile(blank_slice, (n_blank_start, 1, 1))
    blank_end = np.tile(blank_slice, (n_blank_end, 1, 1))

    full_mask_stack = np.concatenate([blank_start, mask_stack, blank_end], axis=0)

    return full_mask_stack
    


if __name__ == "__main__":

    astrocytoma_grade = 2      # 2, 3, 4
    plane = "S"                # C, S, T

    mri_stack = load_dcm_slices(astrocytoma_grade, plane)
    
    plot_sliding_slices(mri_stack)

    slice_start = int(input("Enter the number of the first slice where the tumor is visible: "))
    slice_end = int(input("Enter the number of the first slice where the tumor is visible: "))

    mask_stack = manual_segmentation(mri_stack, slice_start, slice_end)
    
    masked_mri = mask_stack*mri_stack

    plot_sliding_slices(masked_mri)

    mask_path = os.path.join("manual labels", "TU " + str(astrocytoma_grade), plane, "mask.npy")
    os.makedirs(os.path.dirname(mask_path), exist_ok=True)
    np.save(mask_path, mask_stack)


    