# GendScan

GendScan is an intelligent decision support tool for detecting image tampering. This project integrates various advanced deep learning models and techniques to provide a robust solution for forensic image analysis. The tool is designed to assist law enforcement agencies, forensic experts, and other stakeholders in identifying manipulated media efficiently.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Technologies Used](#technologies-used)
- [Model Weights](#model-weights)
- [File Structure](#file-structure)
- [Contributing](#contributing)
- [License](#license)

## Features
- **Image Tampering Detection**: Detects various forms of manipulation in images.
- **Metadata Extraction**: Extracts and analyzes metadata from images and videos.
- **User-Friendly Interface**: Intuitive web application interface built with Streamlit.
- **Customizable Models**: Supports different model architectures and configurations for tampering detection.

## Installation
1. Clone the repository:
   ```
   git clone https://github.com/Islemds/GendScan.git
   cd GendScan
   ```
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Download and place the model weights in the designated folder. [Link to Weights](#model-weights)

## Usage
1. Run the Streamlit application:
   ```
   streamlit run app.py
   ```
2. Access the web interface through your browser at `http://localhost:8501`.
3. Upload an image file to perform tampering detection. The results will be displayed in the interface.

## Technologies Used
- **Frontend**: Streamlit
- **Deep Learning**: TensorFlow, Keras, segmentation_models, classification_models
- **Image Processing**: OpenCV, scikit-image

## Model Weights
Due to the size restrictions on GitHub, the model weights are stored externally. You can download the weights from the following link and place them in the project folder:
[Drive link](#) <!-- Replace with actual drive link -->

## File Structure
```
GendScan/
│
├── app.py                   
├── models    
├── binary_image_classification.py
├── build_model.py        
├── src/                  
├── requirements.txt           
└── README.md                 
```

## Contributing
Contributions are welcome! If you have any ideas, suggestions, or issues, feel free to open an issue or create a pull request. Please make sure to follow the contribution guidelines.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
