## Starting the Application

1.Open Terminal/Command Prompt 
2.Navigate to project folder:

## cd path/to/CircuitGuard_Project

## Run the application:
streamlit run app.py

## Access the app: 
Browser opens automatically at http://localhost:8501

## User Interface Overview
Sidebar (Left Panel)

Model Path: Location of the trained model file
Load Model: Button to load the model
Detection Mode: Choose between two modes
About: Information about the system

Main Panel (Center)

Image upload area
Results visualization
Export options


## Configuration
Loading the Model
# Step 1: Locate your model file
Default path: models/efficientnet_state_dict.pkl
# Step 2: Enter the path in sidebar
# Step 3: Click "Load Model"

Wait for green success message: ✅ "Model loaded successfully!"

## Detection Modes
Mode 1: Classification Only
Best for:
Quick testing of individual defects
Pre-cropped defect images
Fast classification needs

## How to use:
# Select "Classification Only" in sidebar
Click "Browse files" or drag & drop image
Supported formats: PNG, JPG, JPEG
Click "🔍 Classify Defect"
# View results:
Predicted defect type
Confidence percentage
Top 3 predictions

# Mode 2: Full Pipeline (with Template)
Best for:
Complete PCB inspection
Template-based comparison
Multiple defect detection
# How to use:
Select "Full Pipeline (with Template)" in sidebar
Upload two images:
Left: Template (defect-free reference PCB)
Right: Test (PCB to inspect)
Click "🔍 Detect Defects"
View results in stages:

Original images
Processing steps (difference map, threshold)
Detected regions with bounding boxes
Classification for each region



## Exporting Results:
Run a prediction first
Scroll to "💾 Export Results"

## Download Prediction Log:
CSV file with predictions

## Defect Type Guide
1. Missing Hole
Description: Holes that should be drilled but are absent
Appearance: Solid area where hole should be
Common causes: Manufacturing error, drilling failure
2. Mouse Bite
Description: Small notches at board edges
Appearance: Semicircular indentations
Common causes: Breakout tab removal
3. Open Circuit
Description: Break in conductive trace
Appearance: Gap in copper traces
Common causes: Etching issues, physical damage
4. Short
Description: Unwanted connection between traces
Appearance: Copper bridge between traces
Common causes: Over-etching, contamination
5. Spur
Description: Extra copper projection
Appearance: Small protrusion from trace
Common causes: Etching irregularities
6. Spurious Copper
Description: Unwanted copper patches
Appearance: Random copper fragments
Common causes: Incomplete etching



