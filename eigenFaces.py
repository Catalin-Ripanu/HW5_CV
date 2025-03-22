import os
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


@dataclass
class Space:
    """Represents a face space containing data for face reconstruction."""
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    mean: np.ndarray


def load_images(dir_path: str) -> Dict[str, List[np.ndarray]]:
    """Load and parse face images from the given directory."""
    images = defaultdict(list)
    for filename in os.listdir(dir_path):
        img_path = os.path.join(dir_path, filename)
        img = cv.cvtColor(cv.imread(img_path), cv.COLOR_BGR2GRAY)
        
        # Extract person_id from filename (format: person_id_illumination_id.jpg)
        person_id = filename.split('_')[0]
        images[person_id].append(img)
    
    return images


def learn_space(images: List[np.ndarray], num_components: int) -> Space:
    """Learn the eigenface space for a set of images."""
    # Flatten images and stack them
    flattened = np.array([img.ravel() for img in images])
    mean_face = flattened.mean(axis=0)
    
    # Center the data
    centered = flattened - mean_face
    
    # Compute eigenvectors using SVD for efficiency
    A = centered.T
    _, s, vh = np.linalg.svd(A.T @ A)
    eigenvectors = (A @ vh.T)[:, :num_components]
    
    # Normalize eigenvectors
    eigenvectors /= np.linalg.norm(eigenvectors, axis=0)
    
    return Space(
        eigenvalues=s[:num_components],
        eigenvectors=eigenvectors,
        mean=mean_face
    )


def reconstruct(image: np.ndarray, space: Space) -> np.ndarray:
    """Reconstruct an image using the given face space."""
    vec = image.ravel()
    weights = space.eigenvectors.T @ (vec - space.mean)
    reconstruction = space.eigenvectors @ weights + space.mean
    return reconstruction.reshape(image.shape)


def compute_error(original: np.ndarray, reconstruction: np.ndarray) -> float:
    """Compute reconstruction error between original and reconstructed images."""
    return np.linalg.norm((original.ravel() - reconstruction.ravel()) / 255)


def train(train_path: str, num_components: int = 6) -> Dict[str, Space]:
    """Train eigenface spaces for each person."""
    dataset = load_images(train_path)
    return {
        person_id: learn_space(images, num_components)
        for person_id, images in dataset.items()
    }


def predict(image: np.ndarray, spaces: Dict[str, Space]) -> str:
    """Predict the person ID for a given test image."""
    min_error = float('inf')
    best_person = None
    
    for person_id, space in spaces.items():
        reconstructed = reconstruct(image, space)
        error = compute_error(image, reconstructed)
        
        if error < min_error:
            min_error = error
            best_person = person_id
    
    return best_person


def evaluate(spaces: Dict[str, Space], test_path: str) -> float:
    """Evaluate the classifier on test images."""
    test_images = load_images(test_path)
    correct = 0
    total = 0
    
    for true_id, images in test_images.items():
        for image in images:
            predicted_id = predict(image, spaces)
            if predicted_id == true_id:
                correct += 1
            total += 1
    
    return correct / total


def plot_eigenvalue_spectrum(eigenvalues: np.ndarray, save_path: str = None):
    """Plot the eigenvalue spectrum to help choose number of components."""
    plt.figure(figsize=(10, 5))
    cumulative_variance = np.cumsum(eigenvalues) / np.sum(eigenvalues)
    
    plt.subplot(1, 2, 1)
    plt.plot(eigenvalues[:20], 'bo-')
    plt.title('Top 20 Eigenvalues')
    plt.xlabel('Component Index')
    plt.ylabel('Eigenvalue')
    
    plt.subplot(1, 2, 2)
    plt.plot(cumulative_variance[:20], 'ro-')
    plt.title('Cumulative Variance Explained')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Variance Ratio')
    
    if save_path:
        plt.savefig(save_path)
    plt.close()


def visualize_all_reconstructions(spaces: Dict[str, Space], test_images: Dict[str, List[np.ndarray]], 
                                components_list: List[int], save_path: str = None):
    """Visualize face reconstructions for all people with different numbers of components."""
    num_people = len(spaces)
    n_comps = len(components_list) + 1  # +1 for original image
    
    plt.figure(figsize=(2*n_comps, 2*num_people))
    
    for person_idx, (person_id, space) in enumerate(sorted(spaces.items())):
        # Get first test image for this person
        test_image = test_images[person_id][0]
        
        # Original image
        plt.subplot(num_people, n_comps, person_idx * n_comps + 1)
        plt.imshow(test_image, cmap='gray')
        if person_idx == 0:
            plt.title('Original')
        plt.ylabel(f'Person {person_id}')
        plt.axis('off')
        
        # Reconstructions with different numbers of components
        for comp_idx, num_comp in enumerate(components_list, 2):
            reduced_space = Space(
                eigenvalues=space.eigenvalues[:num_comp],
                eigenvectors=space.eigenvectors[:, :num_comp],
                mean=space.mean
            )
            reconstructed = reconstruct(test_image, reduced_space)

            plt.subplot(num_people, n_comps, person_idx * n_comps + comp_idx)
            plt.imshow(reconstructed, cmap='gray')
            if person_idx == 0:
                plt.title(f'{num_comp} comp.')
            plt.axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()


def analyze_dimensionality_effect(train_path: str, test_path: str, 
                                max_components: int = 20) -> Dict[int, float]:
    """Analyze the effect of dimensionality on classification accuracy."""
    results = {}
    for n_components in range(1, max_components + 1):
        spaces = train(train_path, num_components=n_components)
        accuracy = evaluate(spaces, test_path)
        results[n_components] = accuracy
    return results


def plot_dimensionality_analysis(results: Dict[int, float], save_path: str = None):
    """Plot the relationship between number of components and accuracy."""
    components = list(results.keys())
    accuracies = list(results.values())
    
    plt.figure(figsize=(10, 5))
    plt.plot(components, accuracies, 'go-')
    plt.title('Classification Accuracy vs Number of Components')
    plt.xlabel('Number of Components')
    plt.ylabel('Classification Accuracy')
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    plt.close()



# Update the main function to use the new visualization
def main():
    """Main function to train, evaluate, and analyze the eigenfaces classifier."""
    train_path = os.path.join(".", "Faces", "Train")
    test_path = os.path.join(".", "Faces", "Test")
    
    # Create output directory for plots
    output_dir = Path("./output")
    output_dir.mkdir(exist_ok=True)
    
    # Analyze dimensionality effect
    print("Analyzing dimensionality effect...")
    results = analyze_dimensionality_effect(train_path, test_path)
    plot_dimensionality_analysis(results, output_dir / "dimensionality_analysis.png")
    
    # Train with optimal number of components
    optimal_components = max(results.items(), key=lambda x: x[1])[0]
    print(f"\nOptimal number of components: {optimal_components}")
    
    spaces = train(train_path, num_components=optimal_components)
    
    # Plot eigenvalue spectrum for one person
    first_person_space = next(iter(spaces.values()))
    plot_eigenvalue_spectrum(first_person_space.eigenvalues, 
                           output_dir / "eigenvalue_spectrum.png")
    
    # Visualize reconstructions for all people
    test_images = load_images(test_path)
    visualize_all_reconstructions(spaces, test_images, 
                                [1, 3, 5, 10], 
                                output_dir / "all_reconstructions.png")
    
    # Final evaluation
    accuracy = evaluate(spaces, test_path)
    print(f"\nFinal classification accuracy: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()
