from utils import *
from visualisation import *
from blood_vessels_segmentation import *

import os
import matplotlib.pyplot as plt
import csv
from scipy import ndimage
from skimage import measure, feature, filters


if __name__ == '__main__':


    plane = 'S'             # plane = {S, C, T}

    idxs = [50, 39, 43]     # idxs = [TU_2_slice_idx, TU_3_slice_idx, TU_4_slice_idx]

    idx_ranges = [[46, 55], [35, 48], [36, 53]]
    
    # Numericki rezultati se cuvaju u .csv datotetku
    csv_path = os.path.join('results','{}.csv'.format(plane))

    if os.path.exists(csv_path):
        os.remove(csv_path)

    file = open(csv_path, mode='a', newline='')
    writer = csv.writer(file)
    writer.writerow(['area','total area','perimeter','edges length','contrast','homogenity','energy','entropy'])


    #fig, ax = plt.subplots(1,3,figsize=(8,4))
    #fig0, ax0 = plt.subplots(1,3,figsize=(8,4))
    fig1, ax1 = plt.subplots(1,3,figsize=(8,4))
    fig2, ax2 = plt.subplots(1,3,figsize=(8,4))
    fig3, ax3 = plt.subplots(1,3,figsize=(8,4))
    fig4, ax4 = plt.subplots(1,3,figsize=(8,4))
    #fig5, ax5 = plt.subplots(1,3,figsize=(8,4))
    #fig6, ax6 = plt.subplots(1,2,figsize=(8,4))
    fig7, ax7 = plt.subplots(1,3,figsize=(8,4))
    #fig8, ax8 = plt.subplots(1,3,figsize=(8,4))
    fig9, ax9 = plt.subplots(1,3,figsize=(8,4))

    for i in range(3):

        astrocytoma_grade = 2 + i

        mri_stack = load_dcm_slices(astrocytoma_grade, plane)
        mri_slice = mri_stack[idxs[i],:,:]
        print(mri_stack.shape)

        mask_1 = load_mask_stack(astrocytoma_grade, plane, 'manual')
        mask_2 = load_mask_stack(astrocytoma_grade, plane, 'active contour')

        mask = mask_1
        mask_slice = mask[idxs[i],:,:]

        mri_masked = mri_stack*mask
        mri_masked_nan = np.where(mask==1, mri_stack, np.nan)

        mri_masked_slice = mri_masked[idxs[i],:,:]
        mri_masked_slice_nan = mri_masked[idxs[i],:,:]


        """# Prikaz MRI volumena
        if i == 2:
            volume_rendering(mri_stack, slice_spacing=2.2)"""

        # Prikaz MRI slike i segmentisanog tumora

        contours = measure.find_contours(mask_slice)

        """
        # Prikaz MRI slika

        ax8[i].imshow(mri_slice, cmap='gray')
        ax8[i].set_title('TU {}'.format(astrocytoma_grade))
        ax8[i].set_axis_off()"""


        """
        # Prikaz MRI slika i manuelno segmentisane konture tumora

        ax[i].imshow(mri_slice, cmap='gray')
        for contour in contours:
            ax[i].plot(contour[:, 1], contour[:, 0], linewidth=1, color='red')
        ax[i].set_title('TU {}'.format(astrocytoma_grade))
        ax[i].set_axis_off()
        ax[i].set_xlim(*x_lim)
        ax[i].set_ylim(*y_lim)"""


        # Odredjivanje pozicije tumora i njegove povrsine

        areas = []
        perimeters = []
        for j in range(idx_ranges[i][0], idx_ranges[i][1]+1):

            mask_slice_j = mask[j,:,:]
            mri_masked_j = mri_masked[j,:,:]

            mask_props = measure.regionprops(measure.label(mask_slice_j), mri_masked_j)

            for prop in mask_props:

                if j == idxs[i]:
                    minr, minc, maxr, maxc = prop.bbox
                    x_mid = (maxc + minc)//2
                    x_lim = (x_mid - 50, x_mid + 50)
                    y_mid = (maxr + minr)//2
                    y_lim = (y_mid + 50, y_mid - 50)

                areas.append(prop.area)
                perimeters.append(prop.perimeter)
            
        area = np.mean(areas)
        perimeter = np.mean(perimeters)

        area_total = np.sum(np.where(mask.flatten(),1,0))

        """
        # Prikaz histograma tumora

        ax0[i].hist(mri_masked_nan.flatten(), density=True, range = (0,100), bins=20)
        ax0[i].set_title('TU {}'.format(astrocytoma_grade))
        ax0[i].set_xlabel('Intenzitet piksela')
        ax0[i].set_ylabel('Gustina verovatnoće')

        # Prikaz histograma samo za TU 2 i TU 3

        if astrocytoma_grade != 4:
            ax6[i].hist(mri_masked_nan.flatten(), density=True, range = (10,40), bins=20)
            ax6[i].set_title('TU {}'.format(astrocytoma_grade))
            ax6[i].set_xlabel('Intenzitet piksela')
            ax6[i].set_ylabel('Gustina verovatnoće')

        # Prikaz boxplot-a tumora

        #data = [x for x in mri_masked_slice_nan.flatten() if np.isfinite(x)]   # 1 slajs
        data = [x for x in mri_masked_nan.flatten() if np.isfinite(x)]          # svi slajsevi

        flierprops = dict(marker='o', markerfacecolor='black', markersize=1, linestyle='none')
        ax5[i].boxplot(data, showmeans=True, flierprops=flierprops)
        ax5[i].set_title('TU {}'.format(astrocytoma_grade))
        ax5[i].set_ylabel('Intenzitet piksela')
        """

        # Prikaz rezultata razlicitih metoda segmentacije

        contours_1 = measure.find_contours(mask_1[idxs[i],:,:])
        contours_2 = measure.find_contours(mask_2[idxs[i],:,:])

        ax7[i].imshow(mri_slice, cmap='gray')
        for contour in contours_1:
            ax7[i].plot(contour[:, 1], contour[:, 0], '--', linewidth=1, label='manual')
            break
        for contour in contours_2:
            ax7[i].plot(contour[:, 1], contour[:, 0], linewidth=1, color='red', label='active contour')
            break
        ax7[i].set_title('TU {}'.format(astrocytoma_grade))
        ax7[i].legend()
        ax7[i].set_axis_off()
        ax7[i].set_xlim(*x_lim)
        ax7[i].set_ylim(*y_lim)
        
        # Prikaz magnitude gradijenta

        grad_x = ndimage.sobel(mri_slice, axis=0)
        grad_y = ndimage.sobel(mri_slice, axis=1)

        magnitude = np.hypot(grad_x, grad_y)
        #magnitude = (magnitude / magnitude.max())

        magnitude_masked = magnitude*mask_slice

        cax = ax1[i].imshow(magnitude_masked, cmap='gray')
        ax1[i].set_title('TU {}'.format(astrocytoma_grade))
        fig1.colorbar(cax, ax=ax1[i], orientation='vertical')
        ax1[i].set_axis_off()
        ax1[i].set_xlim(*x_lim)
        ax1[i].set_ylim(*y_lim)

        # Prikaz histograma gradijenta

        magnitude_masked_nan = np.where(mask_slice==0, np.nan, magnitude_masked)

        ax2[i].hist(magnitude_masked_nan.flatten(), density=True)
        ax2[i].set_title('TU {}'.format(astrocytoma_grade))
        ax2[i].set_xlabel('Intenzitet piksela')
        ax2[i].set_ylabel('Broj piksela')

        # Prikaz boxplot-a gradijenta

        data = [x for x in magnitude_masked_nan.flatten() if np.isfinite(x)]
        ax3[i].boxplot(data, showmeans=True)
        ax3[i].set_title('TU {}'.format(astrocytoma_grade))
        ax3[i].set_ylabel('Intenzitet piksela')

        # Canny edge detection za detekciju krvnih sudova

        (vessels_mask, mri_vessels) = blood_vessel_segmentation(mri_stack, mask)
        print(np.unique(vessels_mask))

        vessels_mask_slice = vessels_mask[idxs[i],:,:]
        vessels_mask_slice_nan = np.where(vessels_mask_slice==0,np.nan,vessels_mask_slice)

        vessels_masked = vessels_mask*mri_stack
        vessels_masked_slice = vessels_masked[idxs[i],:,:]

        ax4[i].imshow(vessels_mask_slice, cmap='gray')
        ax4[i].set_title('TU {}'.format(astrocytoma_grade))
        ax4[i].set_axis_off()
        ax4[i].set_xlim(*x_lim)
        ax4[i].set_ylim(*y_lim)

        ax9[i].imshow(mri_slice, cmap='gray')
        ax9[i].imshow(vessels_mask_slice_nan, cmap='gray')
        ax9[i].set_title('TU {}'.format(astrocytoma_grade))
        ax9[i].set_axis_off()
        #ax9[i].set_xlim(*x_lim)
        #ax9[i].set_ylim(*y_lim)


        #volume_rendering(mri_vessels, background=(0,0,0))

        n_pixels = np.sum(vessels_mask.flatten())

        # GLCM (Grey-Level Co-occurrence Matrix)

        mri_bbox = mri_slice[minr:maxr,minc:maxc]

        glcm = feature.graycomatrix(mri_bbox,distances=[1], angles=[0], levels=255, symmetric=True)

        if i == 0:
            glcm_2 = glcm
        elif i == 1:
            glcm_3 = glcm
        else:
            glcm_4 = glcm

        # Racunanje parametara teksture iz kookurensne matrice

        contrast = np.squeeze(feature.graycoprops(glcm, 'contrast'))
        homogeneity = np.squeeze(feature.graycoprops(glcm, 'homogeneity'))
        energy = np.squeeze(feature.graycoprops(glcm, 'energy'))

        entropy = measure.shannon_entropy(mri_bbox)

        # Upisivanje vrednosti u .csv datoteku

        row = [area, area_total, perimeter, n_pixels, contrast, homogeneity, energy, entropy]
        writer.writerow(row)


    #fig.tight_layout()
    #fig0.tight_layout()
    fig1.tight_layout()
    fig2.tight_layout()
    fig3.tight_layout()
    fig4.tight_layout()
    #fig5.tight_layout()
    #fig6.tight_layout()
    fig7.tight_layout()
    #fig8.tight_layout()
    fig9.tight_layout()

    file.close()

    plt.figure(figsize=(8,4))

    plt.subplot(1,3,1)
    plt.imshow(glcm_2[:, :, 0, 0], cmap='gray')
    plt.title('TU 2')
    plt.xlim(0,100)
    plt.ylim(0,100)

    plt.subplot(1,3,2)
    plt.imshow(glcm_3[:, :, 0, 0], cmap='gray')
    plt.title('TU 3')
    plt.xlim(0,100)
    plt.ylim(0,100)

    plt.subplot(1,3,3)
    plt.imshow(glcm_4[:, :, 0, 0], cmap='gray')
    plt.title('TU 4')
    plt.xlim(0,100)
    plt.ylim(0,100)

    plt.tight_layout()

    plt.show()
       

