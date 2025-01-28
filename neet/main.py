import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageTk
import json
from flask import Flask, render_template, request, jsonify
import webbrowser
import csv
from crop3 import OMRMarkerDetector
from samplenee import process_omr_sheet
from datetime import datetime

class OMRProcessorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("NEET OMR Sheet Processor")
        self.root.geometry("1200x800")

        # Instance variables
        self.image_files = []
        self.current_index = 0
        self.answer_key = None
        self.processing = False
        self.detector = OMRMarkerDetector()
        self.template_path = "neettemp.json"
        
        # Create a new CSV file for results
        self.create_new_results_file()

        # Start Flask server in a separate thread
        self.flask_thread = threading.Thread(target=self.run_flask_server)
        self.flask_thread.daemon = True
        self.flask_thread.start()

        self.setup_gui()

    def create_new_results_file(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_results_file = f"neet_results_{timestamp}.csv"
        # Create CSV with headers
        headers = ['Roll Number'] + [f'Q{i}' for i in range(1, 201)] + ['Score', 'Correct Answers', 'Wrong Answers', 'Unattempted']
        with open(self.current_results_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)

    def setup_gui(self):
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Control buttons frame
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))

        # Buttons
        ttk.Button(control_frame, text="Upload OMR Sheet Folder", 
                  command=self.upload_folder).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Input Answer Key", 
                  command=self.open_answer_key_page).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Load Answer Key", 
                  command=self.load_answer_key).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Process OMR Sheet", 
                  command=self.process_current_sheet).pack(side=tk.LEFT, padx=5)

        # Navigation frame
        nav_frame = ttk.Frame(main_frame)
        nav_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(nav_frame, text="Previous", 
                  command=self.previous_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(nav_frame, text="Next", 
                  command=self.next_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(nav_frame, text="Cancel", 
                  command=self.cancel_processing).pack(side=tk.LEFT, padx=5)
        ttk.Button(nav_frame, text="Reset", 
                  command=self.reset_gui).pack(side=tk.LEFT, padx=5)

        # Image display
        self.image_label = ttk.Label(main_frame)
        self.image_label.pack(fill=tk.BOTH, expand=True)

        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(main_frame, variable=self.progress_var, 
                                          maximum=100)
        self.progress_bar.pack(fill=tk.X, pady=(10, 0))

        # Status label
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(main_frame, textvariable=self.status_var)
        self.status_label.pack(pady=(5, 0))

    def run_flask_server(self):
        app = Flask(__name__)

        @app.route('/')
        def index():
            return render_template('answer_key.html')

        @app.route('/submit_answer_key', methods=['POST'])
        def submit_answer_key():
            try:
                data = request.get_json()
                if not data:
                    return jsonify({"status": "error", "message": "No data received"}), 400
                
                with open('answer_key.json', 'w') as f:
                    json.dump(data, f)
                return jsonify({"status": "success"})
            except Exception as e:
                return jsonify({"status": "error", "message": str(e)}), 500

        app.run(port=5000, threaded=True)

    def open_answer_key_page(self):
        webbrowser.open('http://localhost:5000')

    def load_answer_key(self):
        try:
            answer_key_path = 'answer_key.json'
            if not os.path.exists(answer_key_path):
                messagebox.showerror("Error", "Answer key file not found!")
                return
                
            with open(answer_key_path, 'r') as f:
                self.answer_key = json.load(f)
                
            if not self.answer_key:
                messagebox.showerror("Error", "Answer key is empty!")
                return
                
            messagebox.showinfo("Success", "Answer key loaded successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load answer key: {str(e)}")

    def upload_folder(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.image_files = [
                os.path.join(folder_path, f) for f in os.listdir(folder_path)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ]
            if self.image_files:
                self.current_index = 0
                self.display_current_image()
            else:
                messagebox.showwarning("Warning", "No image files found in the selected folder!")

    def display_current_image(self):
        if not self.image_files:
            return

        image_path = self.image_files[self.current_index]
        image = Image.open(image_path)
        
        # Resize image to fit the window while maintaining aspect ratio
        display_size = (800, 600)
        image.thumbnail(display_size, Image.Resampling.LANCZOS)
        
        photo = ImageTk.PhotoImage(image)
        self.image_label.configure(image=photo)
        self.image_label.image = photo

    def process_current_sheet(self):
        if not self.image_files:
            messagebox.showerror("Error", "Please load images first!")
            return

        self.processing = True
        threading.Thread(target=self._process_sheet).start()

    def _process_sheet(self):
        try:
            image_path = self.image_files[self.current_index]
            
            # Update status
            self.status_var.set("Detecting markers...")
            self.progress_var.set(25)

            # Detect markers and correct perspective
            result_image, markers = self.detector.detect_markers(image_path)
            if len(markers) != 4:
                raise Exception("Failed to detect all markers")

            corrected = self.detector.correct_perspective(cv2.imread(image_path), markers)
            if corrected is None:
                raise Exception("Failed to correct perspective")

            # Save corrected image
            temp_path = "temp_corrected.jpg"
            cv2.imwrite(temp_path, corrected)

            # Update status
            self.status_var.set("Processing OMR sheet...")
            self.progress_var.set(50)

            # Process the OMR sheet using the function from samplenee.py
            roll_number, section_ranges, section_answers, output_image = process_omr_sheet(temp_path, self.template_path)
            
            if roll_number is None or section_answers is None:
                raise Exception("Failed to process OMR sheet")

            # Update status
            self.status_var.set("Generating results...")
            self.progress_var.set(75)

            # Prepare all answers in sequence
            all_answers = []
            for section_name in sorted(section_answers.keys()):
                all_answers.extend(section_answers[section_name])

            # Calculate score if answer key is available
            score = 0
            if self.answer_key:
                correct_answers = 0
                wrong_answers = 0
                unattempted = 0
                
                for i, (student_ans, correct_ans) in enumerate(zip(all_answers, self.answer_key.values())):
                    if student_ans == "No answer" or student_ans == "multiple":
                        unattempted += 1
                    elif student_ans == correct_ans:
                        correct_answers += 1
                    else:
                        wrong_answers += 1
                
                # Calculate score: +4 for correct, -1 for wrong
                score = (correct_answers * 4) - (wrong_answers * 1)

            # Prepare row for CSV
            row = [roll_number]
            row.extend(all_answers)
            row.extend([score, correct_answers, wrong_answers, unattempted])

            # Append to CSV file
            with open(self.current_results_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)

            # Show results dialog
            if self.answer_key:
                messagebox.showinfo("Results", 
                                  f"Roll Number: {roll_number}\n"
                                  f"Total Score: {score}\n"
                                  f"Correct Answers: {correct_answers}\n"
                                  f"Wrong Answers: {wrong_answers}\n"
                                  f"Unattempted: {unattempted}")
            else:
                messagebox.showinfo("Results", 
                                  f"Roll Number: {roll_number}\n"
                                  "Answer key not loaded - no score calculated")

            # Cleanup
            if os.path.exists(temp_path):
                os.remove(temp_path)

            self.status_var.set("Processing complete!")
            self.progress_var.set(100)

            # Display processed image if available
            if output_image is not None:
                processed_img = Image.fromarray(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
                processed_img.thumbnail((800, 600), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(processed_img)
                self.image_label.configure(image=photo)
                self.image_label.image = photo

        except Exception as e:
            messagebox.showerror("Error", f"Processing failed: {str(e)}")
            self.status_var.set("Processing failed!")
            self.progress_var.set(0)

        finally:
            self.processing = False

    def previous_image(self):
        if self.image_files and self.current_index > 0:
            self.current_index -= 1
            self.display_current_image()

    def next_image(self):
        if self.image_files and self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            self.display_current_image()

    def cancel_processing(self):
        if self.processing:
            self.processing = False
            self.status_var.set("Processing cancelled")
            self.progress_var.set(0)

    def reset_gui(self):
        # Create a new results file
        self.create_new_results_file()

        # Reset all variables
        self.image_files = []
        self.current_index = 0
        self.answer_key = None
        self.processing = False
        self.status_var.set("Ready")
        self.progress_var.set(0)
        self.image_label.configure(image='')
        self.image_label.image = None

def main():
    root = tk.Tk()
    app = OMRProcessorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()