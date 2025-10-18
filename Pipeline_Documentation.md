Initial Image Processing Pipeline

Objective - To preprocess and compare a test PCB image against a "golden" template to isolate potential manufacturing defects.
  
Step 1: Image Alignment
The test image is aligned with the template to correct for any minor shifts or rotations during image capture.
This is done using the ORB feature detection algorithm to find and match key points between the two images.

Step 2: Image Subtraction
An absolute difference is calculated between the template and the newly aligned test image.
This step effectively removes all matching areas, leaving behind only the pixels that differ.

Output: Difference Map
The result is a "Difference Map"—a mostly black image where any bright pixels or spots represent the locations of potential defects.
