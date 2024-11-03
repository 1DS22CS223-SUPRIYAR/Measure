import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import skimage as ski
import scipy as sp
import pandas as pd

def main():
    app_body()

def app_body():
    data = None
    st.markdown("<h1 style = 'text-align: center;'>Garment Measurement & Reference Analyzer</h1>" , unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    col3, col4 = st.columns([1, 1])
    with col3:
        if uploaded_file:
            image = ski.io.imread(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)


    col1, col2 = st.columns([1, 1])   
    with col1:   
        if st.button("Measure") and uploaded_file is not None:
            data, plot_data = Process.measure(uploaded_file) 
    with col2:
        if st.button("Clear"):
            uploaded_file = None
            st.rerun()
    if data is not None:
        st.write("### Measurement Comparison Table")
        st.table(data)
        Process.plot_tshirt_widths(image, plot_data['tshirt_crop'], plot_data['height_reference_indices'],
                                            plot_data['height_reference_percentages'], plot_data['width_indices_list'])

    return None



class Process:

    minr, minc, maxr, maxc = None, None, None, None

    @staticmethod
    def measure(filename):
        #Reading Image
        image = ski.io.imread(filename)
        #Grayscale image
        gray_image = ski.color.rgb2gray(image)
        #Generating Binary Mask
        threshold_value = ski.filters.threshold_otsu(gray_image)
        binary_mask = gray_image > threshold_value
        #Closing Patches
        selem = ski.morphology.disk(10)
        closed_mask = ski.morphology.closing(binary_mask, selem)

        min_size = 500
        filtered_mask = ski.morphology.remove_small_objects(closed_mask, min_size=min_size)

        #Labelling Objects
        labeled_mask, num_labels = sp.ndimage.label(filtered_mask)
        if num_labels != 2:
            return None, None

        #Extracting Objects 
        regions = ski.measure.regionprops(labeled_mask)
        object_areas = {region.label: region.area for region in regions}
        sorted_areas = sorted(object_areas.items(), key=lambda x: x[1])
        coin_label, coin_area = sorted_areas[0]
        shirt_label, shirt_area = sorted_areas[-1]
        coin_region = next(region for region in regions if region.label == coin_label)
        coin_diameter_pixels = min(coin_region.bbox[2] - coin_region.bbox[0],
                           coin_region.bbox[3] - coin_region.bbox[1])
        
        #Reference Value 
        cm_per_pixel = 2.54 / coin_diameter_pixels 

        #Extracting Shirt
        shirt_region = next(region for region in regions if region.label == shirt_label)
        Process.minr, Process.minc, Process.maxr, Process.maxc = shirt_region.bbox
        shirt_crop = filtered_mask[Process.minr:Process.maxr, Process.minc:Process.maxc]

        center_col = shirt_crop[:, shirt_crop.shape[1] // 2]

        height_indices = np.where(center_col == 1)[0]
        tshirt_height_pixels = height_indices[-1] - height_indices[0] if len(height_indices) > 0 else 0

        #Generating reference positions to calculate width
        height_reference_percentages = [0.10, 0.20, 0.40, 0.5, 0.60, 0.80, 0.90]
        height_reference_indices = [int(percentage * tshirt_height_pixels) for percentage in height_reference_percentages]

        widths_pixels = []
        width_indices_list = []
        for row in height_reference_indices:
            row_data = shirt_crop[row, :]
            transitions = np.diff(row_data.astype(int))
            transition_indices = np.where((transitions == 1) | (transitions == -1))[0]
            num_transitions = len(transition_indices)

            if num_transitions == 6:
                third_transition = transition_indices[2]
                third_last_transition = transition_indices[-3]
                width_pixels = third_last_transition - third_transition
                width_indices = [third_transition, third_last_transition]
            else:
                width = np.where(row_data == 1)[0]
                width_pixels = width[-1] - width[0] if len(width) > 0 else 0
                width_indices = [width[-1], width[0]]
            width_indices_list.append(width_indices)
            widths_pixels.append(width_pixels)

        #Estimation in units
        shirt_widths_cm = [round(width * cm_per_pixel, 2) for width in widths_pixels]
        shirt_height_cm = round(tshirt_height_pixels * cm_per_pixel, 2)
        shirt_area_cm = shirt_area*cm_per_pixel*cm_per_pixel

        data = {
            "PARAMETER": ["Area", "H1", "H2", "H3", "H4", "H5", "H6", "H7", "V1"],
            "UNITS": ["SQ CMS", "CMS", "CMS", "CMS", "CMS", "CMS", "CMS", "CMS", "CMS"],
            "MASTER": [3400, 37, 52, 85, 40, 39.5, 43, 47, 64.5],
            "OBSERVED": [shirt_area_cm, shirt_widths_cm[0], shirt_widths_cm[1], shirt_widths_cm[2], shirt_widths_cm[3], shirt_widths_cm[4], shirt_widths_cm[5], shirt_widths_cm[6], shirt_height_cm],
            "DIFFERENCE%": []  
        }

        data["DIFFERENCE%"]  = [(abs(data["MASTER"][i] - data["OBSERVED"][i])/data["MASTER"][i])*100 for i in range(len(data["MASTER"]))]
        df = pd.DataFrame(data)

        plot_data = {
            'tshirt_crop': shirt_crop,
            'height_reference_indices': height_reference_indices,
            'height_reference_percentages': height_reference_percentages,
            'width_indices_list': width_indices_list
        }

        return df, plot_data
    
    @staticmethod
    def plot_tshirt_widths(image, tshirt_crop, height_reference_indices, height_reference_percentages, width_indices_list):
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        
        ax[0].imshow(image)
        ax[0].set_title("Original Image")
        ax[0].add_patch(plt.Rectangle((Process.minc, Process.minr), Process.maxc - Process.minc, Process.maxr - Process.minr, linewidth=2, edgecolor='blue', facecolor='none'))
        
        ax[1].imshow(tshirt_crop, cmap='gray')
        ax[1].set_title("T-shirt Cropped Mask with Width Indicators")
        ax[1].axvline(x=tshirt_crop.shape[1] // 2, color='green', linestyle='--', label="Center Column")
        for i, row in enumerate(height_reference_indices):
            ax[1].axhline(y=row, color='red', linestyle='--', label=f"H{i+1}")

            if i < len(width_indices_list) and width_indices_list[i]:
                start_idx, end_idx = width_indices_list[i]
                ax[1].plot(start_idx, row, 'bo')
                ax[1].plot(end_idx, row, 'ro')
                ax[1].plot([start_idx, end_idx], [row, row], color='yellow', linewidth=2, label=f"Width at HR" if i == 0 else "")

        ax[1].legend()
        st.pyplot(fig)


if __name__ == '__main__':
    main()
