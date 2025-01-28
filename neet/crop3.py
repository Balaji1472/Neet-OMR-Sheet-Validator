import cv2
import numpy as np
from typing import List, Tuple, Optional

class OMRMarkerDetector:
    def __init__(self):
        self.output_size = (800, 1000)
        self.threshold_params = [(11, 5), (15, 5), (21, 5), (25, 10), (31, 15)]
        
    def try_multiple_preprocessing(self, image: np.ndarray) -> List[np.ndarray]:
        """Try multiple preprocessing strategies to handle different image conditions."""
        preprocessed_images = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
        
        # Strategy 1: Basic preprocessing
        preprocessed_images.append(self.basic_preprocessing(gray))
        
        # Strategy 2: Contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        preprocessed_images.append(self.basic_preprocessing(enhanced))
        
        # Strategy 3: Sharpening
        sharpened = cv2.filter2D(gray, -1, np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]]))
        preprocessed_images.append(self.basic_preprocessing(sharpened))
        
        return preprocessed_images
    
    def basic_preprocessing(self, gray: np.ndarray) -> np.ndarray:
        """Basic preprocessing pipeline."""
        # Normalize
        normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        
        # Apply bilateral filter
        filtered = cv2.bilateralFilter(normalized, 9, 75, 75)
        
        # Try multiple threshold parameters
        binary_images = []
        for block_size, C in self.threshold_params:
            binary = cv2.adaptiveThreshold(
                filtered,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                block_size,
                C
            )
            binary_images.append(binary)
        
        # Combine binary images
        combined = np.zeros_like(binary_images[0])
        for binary in binary_images:
            combined = cv2.bitwise_or(combined, binary)
            
        # Clean up
        kernel = np.ones((3,3), np.uint8)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
        
        return combined

    def find_potential_markers(self, binary: np.ndarray, image_shape: Tuple[int, int]) -> List[Tuple[int, int, float]]:
        """Find potential markers using contour analysis."""
        height, width = image_shape[:2]
        min_area = (width * height * 0.0001)  # Reduced minimum area
        max_area = (width * height * 0.01)    # Increased maximum area
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        potential_markers = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area or area > max_area:
                continue
                
            # Get bounding box and calculate properties
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w)/h
            rect_area = w * h
            extent = float(area)/rect_area
            
            # More lenient criteria for initial detection
            if (0.5 <= aspect_ratio <= 2.0 and  # More flexible aspect ratio
                0.4 <= extent <= 1.0):          # More flexible extent
                
                # Calculate marker score based on multiple factors
                squareness = 1 - abs(1 - aspect_ratio)
                fullness = extent
                size_score = 1 - abs((area - (min_area * 4)) / (max_area - min_area))
                
                score = (squareness * 0.4 + fullness * 0.4 + size_score * 0.2)
                
                center_x = x + w//2
                center_y = y + h//2
                potential_markers.append((center_x, center_y, score))
        
        return potential_markers

    def detect_markers(self, image_path: str) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """Enhanced marker detection with multiple strategies."""
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Could not read image")
        
        height, width = image.shape[:2]
        best_markers = []
        best_score = 0
        
        # Try multiple preprocessing strategies
        preprocessed_images = self.try_multiple_preprocessing(image)
        
        for binary in preprocessed_images:
            potential_markers = self.find_potential_markers(binary, image.shape)
            
            if len(potential_markers) >= 4:
                # Find corners
                corners = [(0, 0), (width, 0), (0, height), (width, height)]
                current_markers = []
                corner_scores = []
                
                # Find best marker for each corner
                for corner in corners:
                    if potential_markers:
                        # Calculate normalized distances to corner
                        distances = [np.linalg.norm(np.array(corner) - np.array(m[:2])) / np.sqrt(width**2 + height**2) 
                                   for m in potential_markers]
                        # Combine distance and marker score
                        scores = [(1 - d) * m[2] for d, m in zip(distances, potential_markers)]
                        best_idx = np.argmax(scores)
                        current_markers.append(potential_markers[best_idx][:2])
                        corner_scores.append(scores[best_idx])
                        potential_markers.pop(best_idx)
                
                # Calculate overall configuration score
                if len(current_markers) == 4:
                    config_score = np.mean(corner_scores)
                    if config_score > best_score:
                        best_score = config_score
                        best_markers = current_markers
        
        # Visualize results
        result_image = image.copy()
        if best_markers:
            for marker in best_markers:
                x, y = map(int, marker)
                cv2.circle(result_image, (x, y), 10, (0, 0, 255), -1)
                cv2.rectangle(result_image, (x-20, y-20), (x+20, y+20), (0, 255, 0), 2)
        
        return result_image, best_markers


    def correct_perspective(self, image: np.ndarray, markers: List[Tuple[int, int]]) -> Optional[np.ndarray]:
        """Simplified perspective correction without handling rotation."""
        if len(markers) != 4:
            return None

        # Use markers as-is, assuming they are provided in the correct order.
        markers = np.array(markers, dtype=np.float32)

        # Destination points for perspective transform
        dst_points = np.array([
            [0, 0],
            [self.output_size[0], 0],
            [0, self.output_size[1]],
            [self.output_size[0], self.output_size[1]]
        ], dtype=np.float32)

        # Apply perspective transform
        matrix = cv2.getPerspectiveTransform(markers, dst_points)
        corrected = cv2.warpPerspective(image, matrix, self.output_size)

        return corrected



def main():
    detector = OMRMarkerDetector()
    image_path = "neetsr.jpg"  # Replace with your image path
    
    try:
        result_image, markers = detector.detect_markers(image_path)
        if len(markers) == 4:
            print("Successfully detected all 4 markers")
            corrected = detector.correct_perspective(cv2.imread(image_path), markers)
            if corrected is not None:
                cv2.imwrite("corrected_omr.jpg", corrected)
        else:
            print(f"Only found {len(markers)} markers")
            
        cv2.imwrite("detected_markers.jpg", result_image)
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()