from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import cv2
import os
import io
import base64
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon
import warnings
warnings.filterwarnings('ignore')


# Set matplotlib to use non-interactive backend
plt.switch_backend('Agg')

app = Flask(__name__)

CORS(app, origins=["https://third-eye-xi.vercel.app"])# Enable CORS for all routes



class TextDetectionInference:
    """
    Inference class for trained text detection model
    """
    def __init__(self, model_path, input_size=(512, 512)):
        self.input_size = input_size
        self.output_size = (input_size[0]//4, input_size[1]//4)  # Due to stride 4
        
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
    
    def process_image_and_return_highlighted(self, image, threshold=0.5):
        """Process image and return highlighted image as base64"""
        try:
            # Get predictions
            score_map, geometry_map, processed_image, orig_size = self.predict(image, threshold)
            
            # Create highlighted image
            highlighted_image = image.copy()
            
            # Resize binary score map to original image dimensions
            binary_score = (score_map[:, :, 0] > threshold).astype(np.uint8)
            binary_score_resized = cv2.resize(binary_score, (orig_size[1], orig_size[0]), interpolation=cv2.INTER_NEAREST)
            
            # Create a colored overlay from the resized binary score map
            highlight_color = np.array([255, 255, 0], dtype=np.uint8)  # Yellow
            overlay = np.zeros_like(image, dtype=np.uint8)
            overlay[binary_score_resized == 1] = highlight_color
            
            # Blend the overlay with the original image
            alpha = 0.4  # Transparency factor
            cv2.addWeighted(overlay, alpha, highlighted_image, 1 - alpha, 0, highlighted_image)
            
            # Convert BGR to RGB for proper display
            highlighted_image_rgb = cv2.cvtColor(highlighted_image, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            pil_image = Image.fromarray(highlighted_image_rgb)
            
            # Convert to base64
            buffer = io.BytesIO()
            pil_image.save(buffer, format='PNG')
            buffer.seek(0)
            
            # Encode to base64
            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            # Count detected text regions
            text_regions_count = np.sum(binary_score_resized == 1)
            
            return {
                'success': True,
                'highlighted_image': f'data:image/png;base64,{img_base64}',
                'text_regions_detected': int(text_regions_count),
                'confidence_threshold': threshold
            }
            
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }

# Initialize the detector (you'll need to update the model path)
MODEL_PATH = 'best_text_detector.h5' # Update this path
detector = None

try:
    if os.path.exists(MODEL_PATH):
        detector = TextDetectionInference(MODEL_PATH, input_size=(512, 512))
        print("Text detector initialized successfully globally!") # Use print for logs initially
    else:
        print(f"Error: Model file not found at {MODEL_PATH}")
except Exception as e:
    print(f"FATAL ERROR: Failed to initialize detector globally: {str(e)}")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': detector is not None
    })

@app.route('/detect-text', methods=['POST'])
def detect_text():
    """Main endpoint to process uploaded images"""
    global detector
    
    if detector is None:
        return jsonify({
            'success': False,
            'error': 'Text detection model not loaded'
        }), 500
    
    try:
        # Check if file is present in request
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No file uploaded'
            }), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400
        
        # Check file type
        if not file.content_type.startswith('image/'):
            return jsonify({
                'success': False,
                'error': 'Only image files are supported'
            }), 400
        
        # Read and convert image
        file_bytes = file.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({
                'success': False,
                'error': 'Could not decode image'
            }), 400
        
        # Process image with fixed threshold of 0.5
        result = detector.process_image_and_return_highlighted(image, threshold=0.5)
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Error in detect_text endpoint: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }), 500

@app.route('/model-info', methods=['GET'])
def model_info():
    """Get information about the loaded model"""
    global detector
    
    if detector is None:
        return jsonify({
            'model_loaded': False,
            'error': 'Model not loaded'
        })
    
    return jsonify({
        'model_loaded': True,
        'input_size': detector.input_size,
        'output_size': detector.output_size,
        'model_path': MODEL_PATH
    })

@app.errorhandler(413)
def too_large(e):
    return jsonify({
        'success': False,
        'error': 'File too large. Maximum size is 16MB.'
    }), 413

@app.errorhandler(500)
def internal_error(e):
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500

if __name__ == '__main__':
    print("=== Text Detection Flask Backend ===")
    print("Initializing text detection model...")
    
    # Try to initialize the detector
    if detector is not None: # Check if model was loaded globally

        print("✅ Model loaded successfully!")
        print("\n🚀 Starting Flask server...")
        print("📡 Backend will be available at: http://localhost:5000")
        print("🔗 Frontend should send POST requests to: http://localhost:5000/detect-text")
        print("\nAvailable endpoints:")
        print("  - GET  /health      : Health check")
        print("  - POST /detect-text : Process image for text detection")
        print("  - GET  /model-info  : Get model information")
        
        # Configure Flask app
        app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
        
        # Start the server

    else:
        print("❌ Failed to load model. Please check:")
        print(f"   - Model file exists at: {MODEL_PATH}")
        print("   - Model file is valid and not corrupted")
        print("   - You have sufficient memory to load the model")
        print("\n💡 Update MODEL_PATH variable if your model is in a different location.")