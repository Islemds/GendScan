import streamlit as st
from PIL import Image
import numpy as np


from src.MethodeConventionnels.contrast import clahe_contrast  # Assume this imports CLAHE contrast as an example
from src.MethodeConventionnels.EdgeDetection import canny_edge_detection, apply_edge_detection
from src.MethodeConventionnels.ImageDenoisingLinear import gaussian_filter, median_filter
from src.MethodeConventionnels.rotate import rotate_image
from src.MethodeConventionnels.flip import image_flipping
from src.MethodeConventionnels.ColorSpaces import HSL, LAB, HSV, XYZ, YCbCr, rgb2hsv
from src.MethodeConventionnels.Metadata import extract_metadata_from_image

from src.MethodesStatistiques.ELA import ela
from src.MethodesStatistiques.JPEG_GHOST import jpeg_ghost_multiple
from src.MethodesStatistiques.CFA import cfa_tamper_detection
from src.MethodesStatistiques.NoiseInconsistencies import noise_inconsistencies
from src.MethodesStatistiques.NoiseResidualsMedianFiltering import median_filter_and_residuals

from binary_image_classification import model_functions, model_names, predict_on_image, aggregate_predictions

def main():
    # Set a larger title using HTML
    st.markdown("<h1 style='font-size: 3.5em; color: #003366;'>GendScan</h1>", unsafe_allow_html=True)

    # Add a descriptive paragraph
    st.markdown("""
    <p style="font-size: 1.2em; color: #333333;">
    **GendScan**: Une application intelligente d’aide à la décision, conçue pour détecter efficacement les manipulations d’images en analysant leur contenu à l’aide de techniques basiques telles que le traitement d’images et les méthodes statistiques, ainsi que des techniques avancées de deep learning.
    </p>
    """, unsafe_allow_html=True)

    # Upload image
    uploaded_file = st.file_uploader("Choisissez une image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Image Importée", use_column_width=True)

        # Menu bar for selecting functions
        st.sidebar.markdown("<h3 style='color: #003366;'>Function</h3>", unsafe_allow_html=True)
        method = st.sidebar.selectbox("Choisissez une méthode", 
                                      ["Méthodes Conventionnelles", "Méthodes Statistiques", "Méthodes Avancées"])

        # Conditional function selection based on the chosen method
        if method == "Méthodes Conventionnelles":
            st.sidebar.markdown("""
            <div style="background-color: #f0f2f6; padding: 15px; border-radius: 5px; margin-bottom: 20px; color: #333333;">
                <strong style="font-size: 1.2em; color: #003366;">Méthodes Conventionnelles :</strong> Ces méthodes englobent des opérations classiques de traitement d’image, telles que l’amélioration du contraste et de la luminosité pour améliorer la qualité visuelle, la détection des contours pour identifier les bords et les formes, ainsi que d’autres transformations géométriques, comme la rotation et le redimensionnement, permettant d’analyser les caractéristiques visuelles fondamentales des images.
            </div>
            """, unsafe_allow_html=True)
            function = st.sidebar.radio("Choisissez une fonction", [
                "Contrast and Brightness Enhancement", 
                "Edge Detection",  
                "Image Denoising", 
                "Metadata", 
                "Rotation", 
                "Flipping", 
                "Color Space"
            ])
            
            


        elif method == "Méthodes Statistiques":
            st.sidebar.markdown("""
            <div style="background-color: #e8f4f8; padding: 15px; border-radius: 5px; margin-bottom: 20px; color: #333333;">
                <strong style="font-size: 1.2em; color: #003366;">Méthodes Statistiques :</strong> Ces techniques reposent sur des approches statistiques pour identifier des anomalies dans l’image, telles que des résidus de bruit, des variations dans la texture, ou des incohérences dans les schémas de compression. Elles permettent de révéler des artefacts invisibles à l’œil nu, souvent laissés par des manipulations numériques ou des traitements d’image spécifiques.
            </div>
            """, unsafe_allow_html=True)
            function = st.sidebar.radio("Choisissez une fonction", [
                "JPEG GHOST", 
                "Noise Residuals Median Filtering", 
                "Noise Inconsistencies", 
                "ELA (Error Level Analysis)", 
                "CFA"
            ])

        elif method == "Méthodes Avancées":
            st.sidebar.markdown("""
            <div style="background-color: #f9f0f0; padding: 15px; border-radius: 5px; margin-bottom: 20px; color: #333333;">
                <strong style="font-size: 1.2em; color: #003366;">Méthodes Avancées : </strong> Ces méthodes font appel à des techniques sophistiquées, telles que la classification. Elles d’analyser les caractéristiques complexes des images, détecter des manipulations subtiles, et automatiser la prise de décision avec une grande précision.
            </div>
            """, unsafe_allow_html=True)
            function = st.sidebar.radio("Choisissez une fonction", [
                "Classification"
            ])
        
        
        # Collect parameters for specific functions    
        params = {}
        # Edge Detction
        if function == "Edge Detection":
            params['min_val'] = st.sidebar.slider("Minimum Threshold", 0, 255, 100)
            params['max_val'] = st.sidebar.slider("Maximum Threshold", 0, 255, 200)
            
        # Image Denoising:
        elif function == "Image Denoising":
            denoising_method = st.sidebar.radio("Choisissez un type de débruitage", ["Gaussian Filter", "Median Filter"])
            params['kernel_size'] = st.sidebar.slider("Kernel Size", 1, 15, 3, step=2)
            if denoising_method == "Gaussian Filter":
                params['sigma'] = st.sidebar.slider("Sigma", 0.1, 10.0, 1.0)
                params['filter_type'] = "Gaussian"
            elif denoising_method == "Median Filter":
                params['filter_type'] = "Median"
                
        # Rotation:
        elif function == "Rotation":
            params['angle'] = st.sidebar.slider("Angle de Rotation", -180, 180, 0)
            
        # Flipping:
        elif function == "Flipping":
            flip_direction = st.sidebar.radio("Direction de retournement", ["Horizontal", "Vertical", "Horizontal + Vertical"])
            if flip_direction == "Horizontal":
                params['flipCode'] = 1
            elif flip_direction == "Vertical":
                params['flipCode'] = 0
            elif flip_direction == "Horizontal + Vertical":
                params['flipCode'] = -1
                
        # Color space:
        elif function == "Color Space":
            color_space = st.sidebar.radio("Choisissez un espace de couleur", ["HSV", "YCbCr", "HSL", "LAB", "XYZ"])
            params['color_space'] = color_space
            
            
        # Statistique methodes :
        elif function == "ELA (Error Level Analysis)":
            params['quality'] = st.sidebar.slider("Qualité de Réenregistrement (JPEG)", 10, 100, 90)
            params['block_size'] = st.sidebar.slider("Taille du bloc ELA", 1, 16, 8)
            
        elif function == "Noise Inconsistencies":
            params['block_size'] = st.sidebar.slider("Taille du bloc", min_value=4, max_value=64, value=8, step=4)

        elif function == "Noise Residuals Median Filtering":
            params['kernel_size'] = st.sidebar.slider("Kernel Size", min_value=3, max_value=31, value=3)
            params['amplification_factor'] = st.number_input("Amplification Factor", min_value=1, max_value=100, value=30)

        # Placeholder: Apply the selected function (need to implement each separately)
        if st.button("Appliquer la Fonction"):
            modified_image = apply_function(image, function, params)
            st.image(modified_image, caption=f"Image avec {function}", use_column_width=True)
            
            
            
            
            
        
            

def apply_function(image, function, params):
    # Implement each function separately in your image_processing.py or here
    if function == "Contrast and Brightness Enhancement":
        return clahe_contrast(image)  # Example usage of an imported function
    elif function == "Edge Detection":
        return apply_edge_detection(image, min_val=params['min_val'], max_val=params['max_val'])
    elif function == "Image Denoising":
        if params['filter_type'] == "Gaussian":
            return gaussian_filter(image, kernel_size=params['kernel_size'], sigma=params['sigma'])
        elif params['filter_type'] == "Median":
            return median_filter(image, kernel_size=params['kernel_size'])
    elif function == "Metadata":
        metadata_info = extract_metadata_from_image(image)
        st.markdown(metadata_info)
        return image  # You'll need to define this function
    elif function == "Rotation":
        return rotate_image(image, angle=params['angle'])
    elif function == "Flipping":
        return image_flipping(image, flipCode=params['flipCode'])
    elif function == "Color Space":
        image_cv = np.array(image)
        if params['color_space'] == "HSV":
            return Image.fromarray(HSV(image_cv))
        elif params['color_space'] == "YCbCr":
            return Image.fromarray(YCbCr(image_cv))
        elif params['color_space'] == "HSL":
            return Image.fromarray(HSL(image_cv))
        elif params['color_space'] == "LAB":
            return Image.fromarray(LAB(image_cv))
        elif params['color_space'] == "XYZ":
            return Image.fromarray(XYZ(image_cv))
        
        
    elif function == "JPEG GHOST":
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        return jpeg_ghost_multiple(image)
    elif function == "Noise Residuals Median Filtering":
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        image_cv = np.array(image)
        result = median_filter_and_residuals(image_cv, kernel_size=params['kernel_size'], amplification_factor = params['amplification_factor'])
        return result
    
    elif function == "Noise Inconsistencies":
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        image_cv = np.array(image)
        noise_result = noise_inconsistencies(image_cv, block_size=params['block_size'])
        return noise_result
    
    elif function == "ELA (Error Level Analysis)":
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        image_cv = np.array(image)
        ela_result = ela(image_cv, quality=params['quality'], block_size=params['block_size'])
        return ela_result
    elif function == "CFA":
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        image_cv = np.array(image)
        cfa_result = cfa_tamper_detection(image_cv)
        return cfa_result
   
   
   
    elif function == "Classification":
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        predictions = []

        for model_func, model_name in zip(model_functions, model_names):
            # Load model
            num_classes = 2  # Replace with your number of classes
            model = model_func(num_classes)
            
            # Predict on the image
            model_prediction = predict_on_image(model, image)  # Pass the NumPy array instead of a file path
            predictions.append(model_prediction)
            
        final_prediction = aggregate_predictions(predictions)
        st.markdown(f"**Predicted Label:** {final_prediction['predicted_label']} with a probabilty {final_prediction['predicted_probability']}")
        return image

if __name__ == "__main__":
    main()
