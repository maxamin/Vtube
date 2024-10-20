## Disclaimer

I am not the original author of this project. This repository is shared purely for **educational purposes**. 
Please use this knowledge responsibly and respect all applicable laws and regulations. The author or contributor of this repository is not responsible for any misuse or damage caused by the use of this code.

# Model Trainer Documentation

## Overview
The Model Trainer is a graphical interface designed to facilitate the training of machine learning models. It streamlines the process of creating new models or loading existing ones while providing enhanced functionality for training efficiency and testing.

## Features

1. **Graphical Training Interface**
   - Create new models or load existing models with ease by selecting the path directly.

2. **Model Parameter Display**
   - Read and display the model parameters for easy reference.

3. **Simplified Parameter Input**
   - Users can select parameters and run training without repeatedly inputting them.

4. **Hyperparameter Adjustment**
   - Adjust hyperparameters, including:
     - Learning Rate
     - Structural Similarity
     - Pixel Value Difference
     - Mouth and Eye Priority

5. **Material Loading Options**
   - Choose whether to load materials from subdirectories.

6. **Training Preview Screen**
   - Preview training visuals in a 5-column layout, supporting models with 256 or more layers, including multiple previews for larger models.

7. **Effect Testing**
   - Test the face-swapping effects using real environment images during training. Supports both images and real-time screenshots, displaying results after merging in a three-column layout.

8. **Loss Management**
   - Increase the training frequency of materials with high loss periodically, with the option to export a loss list.

9. **Training Efficiency Improvement**
   - Reduced iteration time per training session to enhance overall training efficiency.

---

# VtubeKit_Live Documentation

## Overview
VtubeKit_Live is an advanced tool for real-time image processing, particularly in face-swapping applications. It enhances user experience with real-time previews and improved face detection capabilities.

## Features

1. **Real-Time Screen and Window Capture**
   - Capture real-time screen or window content, allowing for immediate preview of processed images while watching movies or browsing the web.

2. **Enhanced Face Detection**
   - Improved face detection speeds through in-situ detection and scaling down the original frame for feature point recalibration.

3. **Face Identity Filtering**
   - Ensure that the face-swapping target remains consistent when multiple faces are present in the frame.

4. **Face Contour Adjustment**
   - Adjust face contours to address significant differences between target and source faces, maintaining similarity with functions like multi-point face slimming.

5. **Real-Time Segmentation Model**
   - Utilize real-time segmentation models for more accurate contour segmentation, especially for occlusions.

6. **Flexible Output Options**
   - Choose between horizontal and vertical layouts for original and merged images. When no face is detected, users can select original or frozen frame outputs.

7. **Convenient Recording Features**
   - Added functionalities for video recording and image saving, including audio recording capabilities during video capture.

8. **Model Format with Activation**
   - Employ the VTFM model format, which allows the inclusion of authorization information. Users must enter an activation code to protect the creator's work.

---

## Conclusion
The Model Trainer and VtubeKit_Live tools enhance the efficiency and usability of training machine learning models and real-time image processing. Their features streamline workflows and improve the quality of outcomes for users engaged in face-swapping and related tasks.
