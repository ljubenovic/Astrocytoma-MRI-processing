import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import vtk
from utils import *

def plot_sliding_slices(slices, idx_start=None, idx_end=None):

    if idx_end:
        slices = slices[:idx_end+1,:,:]
    if idx_start:
        slices = slices[idx_start:,:,:]

    def update_image(index):
        ax.imshow(slices[index], cmap='gray')
        ax.set_title(f"Image {index + 1}/{len(slices)}")
        fig.canvas.draw()

    def on_key(event):
        global current_index
        if event.key == 'right':
            if current_index < len(slices) - 1:
                current_index += 1
                update_image(current_index)
        elif event.key == 'left':
            if current_index > 0:
                current_index -= 1
                update_image(current_index)

    global current_index
    current_index = 0

    fig, ax = plt.subplots()
    fig.canvas.mpl_connect('key_press_event', on_key)
    update_image(current_index)
    plt.show()


def volume_rendering(slices, background = (0,0,0), slice_spacing=1.0, change_opacity=False):
        
        n_slices, height, width = slices.shape
        data_matrix = slices.astype(np.uint8)
        
        MIN_VALUE = np.min(data_matrix)
        MAX_VALUE = np.max(data_matrix)
        
        image_data = vtk.vtkImageData()
        image_data.SetDimensions(width, height, n_slices)  # dimensions: (width, height, depth)
        image_data.AllocateScalars(vtk.VTK_UNSIGNED_SHORT, 1)

        image_data.SetSpacing(1.0, 1.0, slice_spacing)

        for z in range(n_slices):
            for y in range(height):
                for x in range(width):
                    image_data.SetScalarComponentFromFloat(x, y, z, 0, data_matrix[z, y, x])

        # Create volume rendering mapper
        volume_mapper = vtk.vtkSmartVolumeMapper()
        volume_mapper.SetInputData(image_data)

        # Create volume actor
        volume = vtk.vtkVolume()
        volume.SetMapper(volume_mapper)

        # Set the color and opacity transfer functions
        color_transfer_function = vtk.vtkColorTransferFunction()
        opacity_transfer_function = vtk.vtkPiecewiseFunction()

        color_transfer_function.AddRGBPoint(MIN_VALUE, 0.0, 0.0, 0.0)       # min intensity is black (0.0, 0.0, 0.0)
        color_transfer_function.AddRGBPoint(0.8*MAX_VALUE, 1.0, 1.0, 1.0)   # max intensity is white (1.0, 1.0, 1.0)

        if change_opacity:
            opacity_transfer_function.AddPoint(MIN_VALUE, 0)
            opacity_transfer_function.AddPoint(int(0.25*MAX_VALUE), 0.02)
            opacity_transfer_function.AddPoint(int(0.6*MAX_VALUE), 0.4)
            opacity_transfer_function.AddPoint(int(0.8*MAX_VALUE), 0.6)
            opacity_transfer_function.AddPoint(MAX_VALUE, 1)

        volume_property = vtk.vtkVolumeProperty()
        volume_property.SetColor(color_transfer_function)
        volume_property.SetScalarOpacity(opacity_transfer_function)
        volume_property.ShadeOff()  # disable shading
        volume_property.SetInterpolationTypeToNearest()

        volume.SetProperty(volume_property)

        # Create a renderer, render window, and interactor
        renderer = vtk.vtkRenderer()
        render_window = vtk.vtkRenderWindow()
        render_window_interactor = vtk.vtkRenderWindowInteractor()

        render_window.AddRenderer(renderer)
        render_window_interactor.SetRenderWindow(render_window)

        # Add the volume to the renderer
        renderer.AddVolume(volume)
        renderer.SetBackground(*background)  # set background color to black

        # Create and set up the interactor style
        style = vtk.vtkInteractorStyleTrackballCamera()
        render_window_interactor.SetInteractorStyle(style)

        # Start the rendering loop
        render_window.Render()
        render_window_interactor.Start()


if __name__ == "__main__":

    idx_ranges = [[46, 55], [35, 48], [36, 53]]

    for i in range(3):

        astrocytoma_grade = 2+i

        mri = load_dcm_slices(astrocytoma_grade, plane='S')
        mask = load_mask_stack(astrocytoma_grade, plane='S', type='manual')

        """volume_rendering(mri, slice_spacing=2.4)
        volume_rendering(mri, slice_spacing=2.4, change_opacity=True)"""
        #volume_rendering(mri*mask, slice_spacing=2.4)
        #volume_rendering(mri*mask, slice_spacing=2.4, background=(0.5,0.5,0.5))
        volume_rendering(mri*mask, slice_spacing=2.4, change_opacity=True)

        from blood_vessels_segmentation import *
        (vessels_mask, mri_vessels) = blood_vessel_segmentation(mri, mask)
        """volume_rendering(vessels_mask*mri, slice_spacing=2.4)
        volume_rendering(vessels_mask, slice_spacing=2.4)"""

        mask1 = np.where(vessels_mask==1,1,mri)
        #plot_sliding_slices(mask1, idx_ranges[i][0], idx_ranges[i][1])

        #volume_rendering(mri_vessels, slice_spacing=2.4, background=(0.5,0.5,0.5))
