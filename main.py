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
        self.root.minsize(1000, 700)
        
        # Set application theme
        style = ttk.Style()
        style.configure("TButton", padding=6, font=('Helvetica', 10, 'bold'))
        style.configure("TFrame", background="#f0f0f0")
        style.configure("TLabel", background="#f0f0f0", font=('Helvetica', 10))
        style.configure("Header.TLabel", font=('Helvetica', 12, 'bold'))
        style.configure("Status.TLabel", font=('Helvetica', 10, 'italic'))
        
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
        # Configure main layout
        self.root.configure(background="#f0f0f0")
        
        # Main container frame
        main_container = ttk.Frame(self.root, padding="20 20 20 20")
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # Title header
        header_frame = ttk.Frame(main_container)
        header_frame.pack(fill=tk.X, pady=(0, 20))
        
        header_label = ttk.Label(header_frame, text="NEET OMR Sheet Processor", 
                                 font=('Helvetica', 16, 'bold'))
        header_label.pack(anchor=tk.CENTER)
        
        # Main content frame with two columns
        content_frame = ttk.Frame(main_container)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left column for controls
        left_frame = ttk.Frame(content_frame, padding="0 0 10 0", width=200)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # Control buttons group with header
        controls_label = ttk.Label(left_frame, text="Controls", style="Header.TLabel")
        controls_label.pack(anchor=tk.W, pady=(0, 10))
        
        # Controls frame with buttons
        controls_frame = ttk.Frame(left_frame)
        controls_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Buttons with consistent width
        button_width = 20
        
        upload_btn = ttk.Button(controls_frame, text="Upload OMR Sheet Folder", 
                               command=self.upload_folder, width=button_width)
        upload_btn.pack(fill=tk.X, pady=5)
        
        answer_key_btn = ttk.Button(controls_frame, text="Input Answer Key", 
                                   command=self.open_answer_key_page, width=button_width)
        answer_key_btn.pack(fill=tk.X, pady=5)
        
        load_key_btn = ttk.Button(controls_frame, text="Load Answer Key", 
                                 command=self.load_answer_key, width=button_width)
        load_key_btn.pack(fill=tk.X, pady=5)
        
        process_btn = ttk.Button(controls_frame, text="Process OMR Sheet", 
                                command=self.process_current_sheet, width=button_width)
        process_btn.pack(fill=tk.X, pady=5)
        
        # Navigation controls
        nav_label = ttk.Label(left_frame, text="Navigation", style="Header.TLabel")
        nav_label.pack(anchor=tk.W, pady=(20, 10))
        
        nav_frame = ttk.Frame(left_frame)
        nav_frame.pack(fill=tk.X)
        
        # Grid for navigation buttons
        prev_btn = ttk.Button(nav_frame, text="← Previous", 
                             command=self.previous_image, width=button_width)
        prev_btn.pack(fill=tk.X, pady=5)
        
        next_btn = ttk.Button(nav_frame, text="Next →", 
                             command=self.next_image, width=button_width)
        next_btn.pack(fill=tk.X, pady=5)
        
        # Utility buttons
        utility_label = ttk.Label(left_frame, text="Utilities", style="Header.TLabel")
        utility_label.pack(anchor=tk.W, pady=(20, 10))
        
        utility_frame = ttk.Frame(left_frame)
        utility_frame.pack(fill=tk.X)
        
        cancel_btn = ttk.Button(utility_frame, text="Cancel Processing", 
                               command=self.cancel_processing, width=button_width)
        cancel_btn.pack(fill=tk.X, pady=5)
        
        reset_btn = ttk.Button(utility_frame, text="Reset All", 
                              command=self.reset_gui, width=button_width)
        reset_btn.pack(fill=tk.X, pady=5)
        
        # Status information
        status_label = ttk.Label(left_frame, text="Status", style="Header.TLabel")
        status_label.pack(anchor=tk.W, pady=(20, 10))
        
        # Status label
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(left_frame, textvariable=self.status_var, 
                                     style="Status.TLabel", wraplength=200)
        self.status_label.pack(fill=tk.X, pady=5)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(left_frame, variable=self.progress_var, 
                                          maximum=100, length=200)
        self.progress_bar.pack(fill=tk.X, pady=5)
        
        # Right column for image display
        right_frame = ttk.Frame(content_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Image display frame with border
        image_frame = ttk.Frame(right_frame, borderwidth=2, relief=tk.GROOVE)
        image_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Image placeholder label with center alignment
        self.image_label = ttk.Label(image_frame, background="#e0e0e0", anchor=tk.CENTER)
        self.image_label.pack(fill=tk.BOTH, expand=True)
        
        # Image index information
        self.image_info_var = tk.StringVar(value="No images loaded")
        self.image_info_label = ttk.Label(right_frame, textvariable=self.image_info_var)
        self.image_info_label.pack(pady=5)
        
        # Set placeholder message
        placeholder_text = "Upload OMR sheet images to begin"
        self.image_label.configure(text=placeholder_text)

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
            self.status_var.set("Answer key loaded")
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
                self.status_var.set(f"Loaded {len(self.image_files)} image(s)")
                self.update_image_info()
            else:
                messagebox.showwarning("Warning", "No image files found in the selected folder!")
                self.status_var.set("No images found in folder")

    def display_current_image(self):
        if not self.image_files:
            return

        image_path = self.image_files[self.current_index]
        try:
            image = Image.open(image_path)
            
            # Get the size of the display area
            self.root.update_idletasks()  # Update layout
            display_width = self.image_label.winfo_width() - 20
            display_height = self.image_label.winfo_height() - 20
            
            # Resize image to fit while maintaining aspect ratio
            img_width, img_height = image.size
            scale = min(display_width/img_width, display_height/img_height)
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            
            # Resize image with high-quality resampling
            resized_img = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(resized_img)
            
            # Display the image
            self.image_label.configure(image=photo, text="")
            self.image_label.image = photo  # Keep a reference
            
            # Update image info
            self.update_image_info()
            
        except Exception as e:
            self.status_var.set(f"Error loading image: {str(e)}")
            self.image_label.configure(image="", text="Error loading image")

    def update_image_info(self):
        if self.image_files:
            filename = os.path.basename(self.image_files[self.current_index])
            self.image_info_var.set(f"Image {self.current_index + 1} of {len(self.image_files)}: {filename}")
        else:
            self.image_info_var.set("No images loaded")

    def process_current_sheet(self):
        if not self.image_files:
            messagebox.showerror("Error", "Please load images first!")
            return

        self.processing = True
        self.status_var.set("Starting processing...")
        self.progress_var.set(10)
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
            correct_answers = 0
            wrong_answers = 0
            unattempted = 0
            
            if self.answer_key:
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

            self.status_var.set(f"Processing complete! Score: {score}")
            self.progress_var.set(100)

            # Display processed image if available
            if output_image is not None:
                processed_img = Image.fromarray(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
                
                # Get the size of the display area
                self.root.update_idletasks()
                display_width = self.image_label.winfo_width() - 20
                display_height = self.image_label.winfo_height() - 20
                
                # Resize image to fit while maintaining aspect ratio
                img_width, img_height = processed_img.size
                scale = min(display_width/img_width, display_height/img_height)
                new_width = int(img_width * scale)
                new_height = int(img_height * scale)
                
                # Resize with high quality
                processed_img = processed_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(processed_img)
                self.image_label.configure(image=photo, text="")
                self.image_label.image = photo

        except Exception as e:
            messagebox.showerror("Error", f"Processing failed: {str(e)}")
            self.status_var.set(f"Processing failed: {str(e)}")
            self.progress_var.set(0)

        finally:
            self.processing = False

    def previous_image(self):
        if self.image_files and self.current_index > 0:
            self.current_index -= 1
            self.display_current_image()
            self.status_var.set(f"Showing image {self.current_index + 1} of {len(self.image_files)}")

    def next_image(self):
        if self.image_files and self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            self.display_current_image()
            self.status_var.set(f"Showing image {self.current_index + 1} of {len(self.image_files)}")

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
        self.image_label.configure(image='', text="Upload OMR sheet images to begin")
        self.image_label.image = None
        self.image_info_var.set("No images loaded")
        messagebox.showinfo("Reset", "Application has been reset. A new results file was created.")

def main():
    root = tk.Tk()
    app = OMRProcessorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()