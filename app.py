from flask import Flask, request, jsonify, render_template
import google.generativeai as genai
import os
from dotenv import load_dotenv
import base64
import json # Import json module

# Load environment variables from .env file
load_dotenv()

# Configure the Google Generative AI API key using GEMINI_API_KEY
API_KEY = os.getenv("GEMINI_API_KEY") # Changed to GEMINI_API_KEY as requested
if not API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set.") # Updated error message
genai.configure(api_key=API_KEY)

# Initialize the Gemini model
# Using 'gemini-2.0-flash-lite' as requested. Please verify this model name
# is available in the Google AI API documentation if you encounter issues.
try:
    model = genai.GenerativeModel('gemini-2.0-flash-lite') # Set model name as requested
except Exception as e:
    print(f"Error initializing model: {e}")
    model = None # Set model to None if initialization fails


app = Flask(__name__)

# Define the prompt for the Gemini model
# This prompt incorporates all the user's specific instructions for waste classification
VERIDIA_VISION_PROMPT = """
You are Veridia Vision, a waste classification assistant.
Analyze the image and identify distinct objects. For each object:
1. Name the specific object (e.g., "plastic bottle", "banana peel", "cardboard box")
2. Classify it as 'recyclable', 'non-recyclable', or 'organic'
3. Classify it according to the rules of Waterloo Region. Provide a brief reason based on typical Waterloo Region guidelines if possible (e.g., "Blue Bin", "Green Bin", "Garbage"). If unsure about specific Waterloo rules for an item, state "Consult Waterloo Region guidelines".
4. Classify humans as 'human'.
5. Classify chips packets, chocolate wrappers, and similar flexible plastic packaging as 'non-recyclable' (Garbage in Waterloo Region).
6. If a bottle has liquid or a container has food residue, classify as 'non-recyclable' (Garbage in Waterloo Region).
7. If you see a person holding an object/objects, ONLY FOCUS ON THE OBJECT(S), not the person(s) or the background, only the objects given!

Provide the output as a JSON array of objects. Each object in the array should have the following keys:
"object_name": The name of the object.
"general_classification": 'recyclable', 'non-recyclable', 'organic', or 'human'.
"waterloo_classification": The classification according to Waterloo Region rules (e.g., "Blue Bin", "Green Bin", "Garbage", "Consult Waterloo Region guidelines", "Human").

Example JSON output format:
[
  {
    "object_name": "plastic water bottle",
    "general_classification": "recyclable",
    "waterloo_classification": "Blue Bin"
  },
  {
    "object_name": "banana peel",
    "general_classification": "organic",
    "waterloo_classification": "Green Bin"
  }
]
If no objects are detected, return an empty JSON array: []
Ensure the output is valid JSON.
"""

@app.route('/')
def index():
    """Renders the main index page."""
    # This route serves the index.html page
    return render_template('index.html') # Assuming your index.html is in a 'templates' folder

@app.route('/camera')
def camera_page():
    """Renders the camera page."""
    # This route serves the camera.html page
    return render_template('camera.html') # Assuming camera.html is in a 'templates' folder

@app.route('/process_frame', methods=['POST'])
def process_frame():
    """Receives image data, sends to Gemini, and returns classification results."""
    if model is None:
        return jsonify({"error": "AI model not initialized. Check API key and model name."}), 500

    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({"error": "No image data provided."}), 400

    image_data_url = data['image']

    # Extract base64 string from data URL (e.g., "data:image/jpeg;base64,...")
    try:
        header, base64_string = image_data_url.split(',', 1)
        image_bytes = base64.b64decode(base64_string)
    except Exception as e:
        print(f"Error decoding image data: {e}")
        return jsonify({"error": "Invalid image data format."}), 400

    # Determine image MIME type from the data URL header
    mime_type = header.split(':')[1].split(';')[0]
    if mime_type not in ['image/jpeg', 'image/png']:
         return jsonify({"error": f"Unsupported image type: {mime_type}. Only JPEG and PNG are supported."}), 415


    try:
        # Prepare the image for the model
        image_part = {
            "mime_type": mime_type,
            "data": image_bytes
        }

        # Send the prompt and image to the model
        response = model.generate_content([VERIDIA_VISION_PROMPT, image_part])

        # Parse the model's response
        # The model is instructed to return JSON, but sometimes it might include
        # extra text or formatting. We need to extract the JSON part.
        # A common issue is markdown code blocks (```json ... ```)
        text_response = response.text.strip()

        # Attempt to find and parse the JSON block
        try:
            # Look for a JSON code block
            if text_response.startswith('```json'):
                json_string = text_response[len('```json'):].strip()
                if json_string.endswith('```'):
                    json_string = json_string[:-len('```')].strip()
            else:
                 # Assume the response is direct JSON if not in a code block
                 json_string = text_response

            # Parse the JSON string
            classification_results = json.loads(json_string)

            # Validate the structure (optional but recommended)
            if not isinstance(classification_results, list):
                 print(f"Warning: Model response is not a JSON array: {text_response}")
                 # Attempt to return an empty list or an error if parsing failed completely
                 return jsonify([]), 200 # Return empty list for unexpected format

            # Further validation of individual items could be added here

            return jsonify(classification_results)

        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            print(f"Raw model response: {text_response}")
            # Return an empty list or an error if JSON parsing fails
            return jsonify([]), 200 # Return empty list for parsing errors
        except Exception as e:
            print(f"Unexpected error processing model response: {e}")
            print(f"Raw model response: {text_response}")
            return jsonify({"error": "Failed to process AI response."}), 500


    except Exception as e:
        print(f"Error generating content from model: {e}")
        return jsonify({"error": "Failed to get response from AI model."}), 500

if __name__ == '__main__':
    # Ensure a 'templates' directory exists and contains index.html and camera.html
    # For production, use a more robust WSGI server like Gunicorn or uWSGI
    # app.run(debug=True) # Use debug=True for development
     app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000))) # For Render deployment
