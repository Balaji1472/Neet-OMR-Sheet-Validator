# NEET OMR Sheet Validator

## Overview

The NEET OMR Sheet Validator is a Python-based application that processes and evaluates OMR sheets using OpenCV techniques. It features a graphical user interface (GUI) built with Tkinter for easy interaction and a Flask-based web interface to input and manage the answer key.

## Features

- **GUI-based OMR Processing**: Upload and process multiple OMR sheets using a user-friendly interface.
- **Automated Marker Detection**: Uses OpenCV to detect OMR sheet markers and apply perspective transformation for accurate analysis.
- **Answer Key Management**: A Flask-based web application to input and store the correct answer key.
- **Automated Score Calculation**: Computes the total score, correct, wrong, and unattempted answers based on the provided answer key.
- **CSV Export**: Saves the processed results into a structured CSV file.

## Installation

### Prerequisites

Ensure you have Python 3 installed on your system.

### Required Dependencies

Install the necessary Python packages using the following command:

```sh
pip install opencv-python numpy pandas pillow flask tkinter
```

## Usage

### Running the Application

1. Clone the repository:
   ```sh
   git clone https://github.com/Balaji1472/Neet-OMR-Validator.git
   cd neet-omr-validator
   ```
2. Run the application:
   ```sh
   python main.py
   ```

### Instructions

1. **Upload OMR Sheets**: Click on "Upload OMR Sheet Folder" to select the folder containing scanned OMR sheets.
2. **Input Answer Key**: Open the answer key input webpage by clicking "Input Answer Key" and submit the correct answers.
3. **Load Answer Key**: Click "Load Answer Key" to import the saved answer key.
4. **Process OMR Sheet**: Click "Process OMR Sheet" to analyze the selected OMR sheet.
5. **View Results**: The results are displayed on the GUI and saved in a CSV file.

## File Structure

- `main.py` - The main application file containing the GUI and processing logic.
- `crop3.py` - Contains functions for OMR marker detection and perspective transformation.
- `samplenee.py` - Handles OMR sheet evaluation.
- `templates/answer_key.html` - Web interface for entering the answer key.
- `answer_key.json` - Stores the correct answers entered via the web interface.

## Technologies Used

- **OpenCV**: Image processing and marker detection.
- **Tkinter**: GUI for user interaction.
- **Flask**: Web application for answer key input.
- **Pandas**: Data handling and CSV export.

## Future Enhancements

- Improve marker detection accuracy.
- Add real-time webcam support for OMR sheet scanning.
- Enhance UI/UX with modern design frameworks.

## License

This project is open-source and available under the MIT License.

## Author

Developed by [Balaji V]

