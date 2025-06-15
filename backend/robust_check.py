import tensorflow as tf
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon
import glob
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

class TextDetectionInference:
    """
    Inference class for trained text detection model
    """
    def __init__(self, model_path, input_size=(512, 512)):
        self.input_size = input_size
        self.output_size = (input_size[0]//4, input_size[1]//4)  # Due to stride 4
        self.output_directory = None # Initialize output directory attribute

        
        # Load the trained model
        print(f"Loading model from {model_path}...")
        self.model = tf.keras.models.load_model(
            model_path,
            custom_objects={
                'dice_loss': self.dice_loss,
                'geometry_loss': self.geometry_loss
            }
        )
        print("Model loaded successfully!")
    
    def dice_loss(self, y_true, y_pred):
        """Dice loss function (needed for loading model)"""
        smooth = 1e-6
        y_true_f = tf.keras.backend.flatten(y_true)
        y_pred_f = tf.keras.backend.flatten(y_pred)
        intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
        return 1 - (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)
    
    def geometry_loss(self, y_true, y_pred):
        """Geometry loss function (needed for loading model)"""
        return tf.keras.losses.huber(y_true, y_pred)
    
    def preprocess_image(self, image):
        """Preprocess image for inference"""
        if isinstance(image, str):
            # If string, assume it's a file path
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Store original dimensions
        orig_h, orig_w = image.shape[:2]
        
        # Resize image
        resized_image = cv2.resize(image, self.input_size)
        normalized_image = resized_image.astype(np.float32) / 255.0
        
        return normalized_image, (orig_h, orig_w)
    
    def predict(self, image_input, threshold=0.5):
        """Make prediction on image"""
        # Preprocess
        processed_image, orig_size = self.preprocess_image(image_input)
        image_batch = np.expand_dims(processed_image, axis=0)
        
        # Predict
        predictions = self.model.predict(image_batch, verbose=0)
        score_map = predictions[0][0]  # Remove batch dimension
        geometry_map = predictions[1][0]  # Remove batch dimension
        
        return score_map, geometry_map, processed_image, orig_size
    
    def extract_text_boxes(self, score_map, geometry_map, threshold=0.5, 
                          nms_threshold=0.3, orig_size=None):
        """Extract text boxes from predictions"""
        # Get pixels above threshold
        text_pixels = np.where(score_map[:, :, 0] > threshold)
        
        if len(text_pixels[0]) == 0:
            return []
        
        boxes = []
        confidences = []
        
        # Extract boxes from high-confidence pixels
        for y, x in zip(text_pixels[0], text_pixels[1]):
            confidence = score_map[y, x, 0]
            
            # Get geometry information
            geom = geometry_map[y, x, :]
            
            # Reconstruct box (simplified - you might want to improve this)
            # This is a basic interpretation of the geometry map
            center_x = x * 4  # Scale back to input size (stride 4)
            center_y = y * 4
            
            # Use geometry map to estimate box size
            width = max(10, geom[2] * self.input_size[1])  # Minimum width of 10
            height = max(10, geom[3] * self.input_size[0])  # Minimum height of 10
            
            # Create box coordinates
            x1 = max(0, center_x - width/2)
            y1 = max(0, center_y - height/2)
            x2 = min(self.input_size[1], center_x + width/2)
            y2 = min(self.input_size[0], center_y + height/2)
            
            boxes.append([x1, y1, x2, y2])
            confidences.append(float(confidence))
        
        if not boxes:
            return []
        
        # Apply Non-Maximum Suppression
        boxes = np.array(boxes)
        confidences = np.array(confidences)
        
        # OpenCV NMS
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(), 
            confidences.tolist(), 
            threshold, 
            nms_threshold
        )
        
        final_boxes = []
        if len(indices) > 0:
            for i in indices.flatten():
                box = boxes[i]
                conf = confidences[i]
                
                # Scale back to original image size if provided
                if orig_size is not None:
                    orig_h, orig_w = orig_size
                    scale_x = orig_w / self.input_size[1]
                    scale_y = orig_h / self.input_size[0]
                    
                    box = [
                        box[0] * scale_x,
                        box[1] * scale_y,
                        box[2] * scale_x,
                        box[3] * scale_y
                    ]
                
                final_boxes.append({
                    'box': box,
                    'confidence': conf
                })
        
        return final_boxes
    
    def visualize_predictions(self, image_input, threshold=0.5, save_path=None):
        """Visualize predictions on image"""
        # Get predictions
        score_map, geometry_map, processed_image, orig_size = self.predict(image_input, threshold)
        
        # Load original image for visualization
        if isinstance(image_input, str):
            orig_image = cv2.imread(image_input)
            orig_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
            original_filename = os.path.basename(image_input)

        else:
            orig_image = image_input.copy()
            original_filename = "detected_image.png" # Default name if input is not a file path

        
        # Extract text boxes (still needed for potential future use or just to return)
        text_boxes = self.extract_text_boxes(score_map, geometry_map, threshold, orig_size=orig_size)
        
        # --- Create and display the original image with highlighted text ---
        highlighted_image = orig_image.copy()
        
        # Resize binary score map to original image dimensions
        binary_score = (score_map[:, :, 0] > threshold).astype(np.uint8)
        binary_score_resized = cv2.resize(binary_score, (orig_size[1], orig_size[0]), interpolation=cv2.INTER_NEAREST)
        
        # Create a colored overlay from the resized binary score map
        highlight_color = np.array([255, 255, 0], dtype=np.uint8) # Yellow
        overlay = np.zeros_like(orig_image, dtype=np.uint8)
        overlay[binary_score_resized == 1] = highlight_color
        
        # Blend the overlay with the original image
        alpha = 0.4 # Transparency factor
        cv2.addWeighted(overlay, alpha, highlighted_image, 1 - alpha, 0, highlighted_image)
        
        # Plot the single image
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        ax.imshow(highlighted_image)
        ax.set_title('Original Image with Highlighted Text')
        ax.axis('off')
        
        plt.tight_layout()
        
        if self.output_directory:
            save_filename = os.path.join(self.output_directory, f"highlighted_{original_filename}")
            plt.savefig(save_filename, dpi=150, bbox_inches='tight')
            print(f"Highlighted image saved to {save_filename}")
        
        plt.show()

        return text_boxes
    
    def test_on_image(self, image_path, threshold=0.5):
        """Test on a single image"""
        print(f"\nTesting on: {os.path.basename(image_path)}")
        
        if not os.path.exists(image_path):
            print(f"Error: Image not found at {image_path}")
            return None
        
        # Run inference and visualization
        detections = self.visualize_predictions(image_path, threshold)
        
        print(f"Found {len(detections)} text regions")
        for i, detection in enumerate(detections):
            box = detection['box']
            conf = detection['confidence']
            print(f"  Box {i+1}: [{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}] confidence: {conf:.3f}")
        
        return detections
    
    def test_on_folder(self, folder_path, threshold=0.5, max_images=10):
        """Test on all images in a folder"""
        print(f"\nTesting on images in folder: {folder_path}")
        
        # Get all image files
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(folder_path, ext)))
            image_files.extend(glob.glob(os.path.join(folder_path, ext.upper())))
        
        if not image_files:
            print("No image files found in the folder!")
            return
        
        print(f"Found {len(image_files)} images")
        
        # Test on subset
        test_images = image_files[:max_images]
        
        results = []
        for img_path in test_images:
            try:
                detections = self.test_on_image(img_path, threshold)
                results.append({
                    'image': img_path,
                    'detections': detections
                })
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
        
        return results
    
    def interactive_test(self):
        """Interactive testing interface"""
        print("\n=== Interactive Text Detection Testing ===")
        print("Choose an option:")
        print("1. Test on single image (provide path)")
        print("2. Test on folder of images")
        print("3. Test on sample images")
        print("4. Exit")
        
        while True:
            choice = input("\nEnter your choice (1-4): ").strip()
            
            if choice == '1':
                image_path = input("Enter image path: ").strip()
                threshold = float(input("Enter confidence threshold (0.1-0.9, default 0.5): ") or "0.5")
                self.test_on_image(image_path, threshold)
                
            elif choice == '2':
                folder_path = input("Enter folder path: ").strip()
                threshold = float(input("Enter confidence threshold (0.1-0.9, default 0.5): ") or "0.5")
                max_images = int(input("Max images to test (default 10): ") or "10")
                self.test_on_folder(folder_path, threshold, max_images)
                
            elif choice == '3':
                self.test_sample_images()
                
            elif choice == '4':
                print("Exiting...")
                break
                
            else:
                print("Invalid choice! Please enter 1-4.")
    
    def test_sample_images(self):
        """Test on some sample images if available"""
        print("\nLooking for sample images...")
        
        # Common sample image locations
        sample_locations = [
            "/kaggle/input/*/",
            "./sample_images/",
            "./test_images/",
            "./"
        ]
        
        found_images = []
        for location in sample_locations:
            images = glob.glob(location + "*.jpg") + glob.glob(location + "*.png")
            found_images.extend(images[:3])  # Take first 3 from each location
        
        if found_images:
            print(f"Found {len(found_images)} sample images")
            for img_path in found_images[:5]:  # Test max 5 images
                self.test_on_image(img_path)
        else:
            print("No sample images found. Please provide image paths manually.")
    
    def batch_test_with_metrics(self, test_images, threshold=0.5):
        """Batch testing with basic metrics"""
        print(f"\nBatch testing on {len(test_images)} images...")
        
        total_detections = 0
        successful_tests = 0
        
        for i, img_path in enumerate(test_images):
            try:
                print(f"\nProcessing {i+1}/{len(test_images)}: {os.path.basename(img_path)}")
                detections = self.visualize_predictions(img_path, threshold)
                total_detections += len(detections)
                successful_tests += 1
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
        
        print(f"\n=== Batch Test Results ===")
        print(f"Successfully processed: {successful_tests}/{len(test_images)} images")
        print(f"Total text regions detected: {total_detections}")
        print(f"Average detections per image: {total_detections/max(successful_tests, 1):.2f}")


def main():
    """Main testing function"""
    print("=== Text Detection Model Tester ===")
    
    # Model path - update this to your trained model
    model_path = "best_text_detector.h5"
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please ensure you have trained the model and the file exists.")
        
        # Try to find model in common locations
        possible_paths = [
            "./best_text_detector.h5",
            "../best_text_detector.h5",
            "/kaggle/working/best_text_detector.h5",
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                print(f"Found model at: {model_path}")
                break
        else:
            print("Could not find model file. Exiting.")
            return
    
    # Initialize inference class
    try:
        detector = TextDetectionInference(model_path, input_size=(512, 512))
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Create output directory if it doesn't exist
    output_dir = "out" # Changed directory name to 'out'
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output images will be saved in: {output_dir}")

    # Pass the output_dir to the detector for use in visualization
    detector.output_directory = output_dir
    
    # Choose testing mode
    print("\nChoose testing mode:")
    print("1. Interactive mode (manual input)")
    print("2. Quick test on sample images")
    print("3. Test on specific image")
    print("4. Test on folder")
    
    mode = input("Enter mode (1-4): ").strip()
    
    if mode == '1':
        detector.interactive_test()
        
    elif mode == '2':
        detector.test_sample_images()
        
    elif mode == '3':
        image_path = input("Enter image path: ").strip()
        threshold = float(input("Enter threshold (default 0.5): ") or "0.5")
        detector.test_on_image(image_path, threshold)
        
    elif mode == '4':
        folder_path = input("Enter folder path: ").strip()
        threshold = float(input("Enter threshold (default 0.5): ") or "0.5")
        max_images = int(input("Max images (default 10): ") or "10")
        detector.test_on_folder(folder_path, threshold, max_images)
        
    else:
        print("Invalid mode. Starting interactive mode...")
        detector.interactive_test()




if __name__ == "__main__":
    main()