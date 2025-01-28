import cv2
import numpy as np
import json
from datetime import datetime
import csv

def load_template(json_path):
    """Load template configuration"""
    with open(json_path, 'r') as f:
        return json.load(f)

def enhance_image(image):
    """
    Enhanced image preprocessing specifically tuned for OMR sheets
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply bilateral filter to reduce noise while preserving edges
    denoised = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    
    # Apply adaptive thresholding with optimized parameters
    binary = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=25,  # Increased block size for better local adaptation
        C=15  # Increased constant for better bubble detection
    )
    
    # Apply morphological operations to clean up the image
    kernel = np.ones((3,3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    return binary

def detect_marked_bubble(binary_image, x, y, bubble_dims, threshold=0.45):
    """
    Improved bubble detection with advanced analysis techniques
    """
    bubble_width, bubble_height = bubble_dims
    half_width, half_height = bubble_width // 2, bubble_height // 2
    
    # Extract ROI with padding
    padding = 2  # Add padding to account for slight misalignments
    roi = binary_image[
        max(0, y - half_height - padding):min(binary_image.shape[0], y + half_height + padding),
        max(0, x - half_width - padding):min(binary_image.shape[1], x + half_width + padding)
    ]
    
    if roi.size == 0:
        return 0.0
    
    # Calculate fill ratio
    total_pixels = roi.size
    filled_pixels = cv2.countNonZero(roi)
    fill_ratio = filled_pixels / float(total_pixels)
    
    # Analyze local contrast
    mean_intensity = np.mean(roi)
    std_intensity = np.std(roi)
    
    # Adjust threshold based on local image characteristics
    local_threshold = threshold
    if std_intensity > 50:  # High contrast area
        local_threshold *= 0.9
    elif std_intensity < 20:  # Low contrast area
        local_threshold *= 1.1
    
    # Check for bubble pattern using contour analysis
    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        contour_area = cv2.contourArea(largest_contour)
        expected_area = np.pi * (min(bubble_width, bubble_height)/2)**2
        area_ratio = contour_area / expected_area
        
        # Analyze contour circularity
        perimeter = cv2.arcLength(largest_contour, True)
        if perimeter > 0:
            circularity = 4 * np.pi * contour_area / (perimeter * perimeter)
        else:
            circularity = 0
        
        # Return confidence score based on multiple factors
        confidence = (
            fill_ratio * 0.4 +  # Fill ratio contribution
            (1.0 - abs(1.0 - area_ratio)) * 0.3 +  # Area ratio contribution
            circularity * 0.3  # Circularity contribution
        )
        
        return confidence if confidence > local_threshold else 0.0
    
    return 0.0

def process_field_block(binary_image, block_config, output_image=None):
    """
    Process a field block with enhanced answer detection
    """
    origin_x, origin_y = block_config['origin']
    h_gap = block_config['bubblesGap']
    v_gap = block_config['labelsGap']
    bubble_dims = block_config.get('bubbleDimensions', [16, 16])
    
    is_vertical = block_config.get('direction', 'horizontal') == 'vertical'
    num_options = len(block_config['bubbleValues'])
    bubble_values = block_config['bubbleValues']
    
    # Parse question range from fieldLabels
    field_label = block_config['fieldLabels'][0]
    if '..' in field_label:
        start_q, end_q = map(int, field_label.split('..'))
        num_questions = end_q - start_q + 1
    else:
        num_questions = len(block_config['fieldLabels'])
    
    results = []
    MARKED_THRESHOLD = 0.5

    for q in range(num_questions):
        marked_options = []
        marked_value = "No answer"  # Default to "No answer"

        for opt in range(num_options):
            if is_vertical:
                x = int(origin_x + (q * h_gap))
                y = int(origin_y + (opt * v_gap))
            else:
                x = int(origin_x + (opt * h_gap))
                y = int(origin_y + (q * v_gap))

            confidence = detect_marked_bubble(binary_image, x, y, bubble_dims)

            if confidence > MARKED_THRESHOLD:
                marked_options.append(bubble_values[opt])

            if output_image is not None:
                color = (0, 255, 0) if confidence > MARKED_THRESHOLD else (0, 0, 255)
                cv2.circle(output_image, (x, y), bubble_dims[0] // 2, color, 2)

        # Determine the final answer based on marked options
        if len(marked_options) > 1:
            marked_value = "multiple"
        elif len(marked_options) == 1:
            marked_value = marked_options[0]
        # else: keeps "No answer" as default

        results.append(marked_value)

    return results

def save_results_to_csv(results_dict, confidence_scores, template_config, filename="omr_results.csv"):
    """
    Save results in a horizontal table format with roll number and 200 questions.
    """
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Create header row for 200 questions
        headers = ["Roll Number"] + [f"Question {i + 1}" for i in range(200)]
        writer.writerow(headers)
        
        # Create data row
        data_row = [results_dict['roll_number']]
        question_index = 1
        
        for section_name, section_config in template_config['fieldBlocks'].items():
            if section_name != 'RollNumber':
                bubble_values = section_config['bubbleValues']
                for marked_opt in results_dict[section_name.lower()]:
                    if question_index > 200:  # Ensure no more than 200 questions
                        break
                    
                    # Store "Multiple" explicitly and skip further checks
                    if marked_opt == "Multiple":
                        data_row.append("Multiple")
                    elif marked_opt != -1:  # Store valid marked option
                        data_row.append(bubble_values[marked_opt])
                    else:  # No answer case
                        data_row.append("No Answer")
                    
                    question_index += 1
        
        # Pad any remaining questions with "No Answer" if less than 200
        while len(data_row) < 201:
            data_row.append("No Answer")
        
        writer.writerow(data_row)



def process_omr_sheet(image_path, template_path):
    """
    Main processing function for NEET OMR sheet
    """
    try:
        # Load template and image
        template_config = load_template(template_path)
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image at {image_path}")

        # Create output image and enhance input image
        output_image = image.copy()
        binary_image = enhance_image(image)

        # Process roll number
        roll_block = template_config['fieldBlocks']['RollNumber']
        roll_digits = process_field_block(binary_image, roll_block, output_image)
        roll_number = ''.join(str(digit) for digit in roll_digits if digit != "No answer")

        # Get question ranges for each section
        section_ranges = {}
        section_answers = {}
        
        for section_name, section_block in template_config['fieldBlocks'].items():
            if section_name != 'RollNumber':
                field_label = section_block['fieldLabels'][0]
                start_q, end_q = map(int, field_label.split('..'))
                section_ranges[section_name] = (start_q, end_q)
                
                # Process section answers
                answers = process_field_block(binary_image, section_block, output_image)
                section_answers[section_name] = answers

        """# Save visualization
        #timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        #output_filename = f"processed_omr_{timestamp}.jpg"
        #cv2.imwrite(output_filename, output_image)

        # Save results to CSV
        csv_filename = f"results_{timestamp}.csv"
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header with actual question numbers
            header = ['Roll Number']
            all_questions = []
            for section_name, (start_q, end_q) in sorted(section_ranges.items()):
                all_questions.extend([f'Q{i}' for i in range(start_q, end_q + 1)])
            header.extend(all_questions)
            writer.writerow(header)
            
             #Write answers
            row = [roll_number]
            for section_name, (start_q, end_q) in sorted(section_ranges.items()):
                row.extend(section_answers[section_name])
            writer.writerow(row)"""

        return roll_number, section_ranges, section_answers, output_image

    except Exception as e:
        print(f"Error in process_omr_sheet: {str(e)}")
        return None, None, None, None


def main():
    """
    Main entry point
    """
    image_path = "corrected_omr.jpg"
    template_path = "neettemp.json"

    try:
        roll_number, section_ranges, section_answers, output_image = process_omr_sheet(image_path, template_path)
        
        if roll_number is not None and section_answers is not None:
            # Display processed image
            cv2.imshow('Processed OMR Sheet', output_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("Failed to process OMR sheet")

    except Exception as e:
        print(f"Error in main: {str(e)}")

if __name__ == "__main__":
    main()