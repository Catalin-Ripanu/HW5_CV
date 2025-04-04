# Face Recognition using Eigenfaces

This project uses Principal Component Analysis (PCA) through the Eigenfaces method to implement facial recognition. It's based on the landmark paper ["Eigenfaces for Recognition" by Turk and Pentland (1991)](http://www.face-rec.org/algorithms/PCA/jcn.pdf).

## Overview

The system classifies images of faces by:
1. Learning separate PCA subspaces for each person in the training set
2. Projecting test images onto these subspaces
3. Classifying based on minimum reconstruction error

## Dataset Structure

The dataset contains facial images of 5 individuals under different lighting conditions:
- Training images: `./Faces/Train/`
- Test images: `./Faces/Test/`

Images are named according to the pattern: `<person_id>_<illumination_id>.jpg`

## Implementation

### Files
- `eigenFaces.py`
  - Main script that runs the entire pipeline
  - Handles training of PCA models for each person
  - Performs classification on test images
  - Extracts features using PCA
  - Reconstructs images from eigenvectors

### Algorithm
1. **Training Phase**
   - For each person, collect all training images
   - Compute mean face and covariance matrix
   - Extract principal components (eigenvectors) and eigenvalues
   - Select top k components based on eigenvalue magnitudes

2. **Testing Phase**
   - For each test image, project onto each person's eigenspace
   - Calculate reconstruction error for each projection
   - Classify as the person with minimum reconstruction error

## Usage

Run the main script to train the classifier and evaluate performance:
```matlab
eigenFaces
```

The script will output the classification rate on test images.

## Dependencies
- MATLAB
- Image Processing Toolbox (recommended)

## Results

The system achieves [classification rate] accuracy on the test set. Performance can be tuned by adjusting the number of principal components used.

## Notes

- The dimensionality reduction aspect of PCA helps manage the "curse of dimensionality"
- Eigenfaces represent the most significant modes of variation in the facial dataset
- The method is particularly sensitive to variations in lighting, expression, and orientation

## References

- Turk, M., & Pentland, A. (1991). Eigenfaces for Recognition. Journal of Cognitive Neuroscience, 3(1), 71-86.
- [JHU Vision Course Handout](http://www.vision.jhu.edu/teaching/vision08/Handouts/case_study_pca1.pdf)
- [Drexel Eigenface Tutorial](http://www.pages.drexel.edu/~sis26/Eigenface%20Tutorial.htm)
