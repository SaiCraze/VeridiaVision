import os
import io
import base64
import json
import time
from flask import Flask, render_template, request, jsonify, redirect, url_for
from PIL import Image, ImageDraw, ImageFont
import google.generativeai as genai
from pydantic import BaseModel, ValidationError # Assuming you might use pydantic later, keeping imports
from dotenv import load_dotenv
import numpy as np
import traceback # Import traceback for better error logging

# --- Configuration & Setup ---
load_dotenv()
app = Flask(__name__)
app.secret_key = os.urandom(24) # Good practice for sessions, even if not used yet

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable not set.")
genai.configure(api_key=api_key)

# Model Names
MODEL_DEFAULT = "gemini-2.0-flash"
MODEL_TEST = "gemini-2.5-flash-preview-0417" # Use the specific preview model ID

# --- Data Model (Optional but good practice) ---
class BoundingBox(BaseModel):
    box_2d: list[int]
    label: str
    object_name: str = "Unknown object" # Default value

# --- Image Processing ---
def draw_bounding_boxes(image_bytes, bounding_boxes):
    """Draws bounding boxes with a theme suitable for dark UI."""
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        width, height = img.size
        draw = ImageDraw.Draw(img)

        # Ensure bounding_boxes is a list of BoundingBox objects if validation happened
        # If validation didn't happen (e.g., error before validation), handle raw dicts
        labels = set()
        valid_box_objects = []
        if bounding_boxes: # Check if list is not None or empty
             for box_data in bounding_boxes:
                 if isinstance(box_data, BoundingBox):
                     labels.add(box_data.label)
                     valid_box_objects.append(box_data)
                 elif isinstance(box_data, dict): # Handle raw dict if validation failed
                     try:
                         # Attempt to create a BoundingBox for consistent access
                         box_obj = BoundingBox(**box_data)
                         labels.add(box_obj.label)
                         valid_box_objects.append(box_obj)
                     except ValidationError:
                         print(f"Skipping drawing invalid box data: {box_data}")
                         continue # Skip drawing this box
                 else:
                     print(f"Skipping drawing unrecognized box data type: {type(box_data)}")
                     continue # Skip drawing this box

        if not valid_box_objects:
             print("No valid boxes to draw.")
             return image_bytes # Return original if no valid boxes

        sorted_labels = sorted(list(labels))

        # Vibrant Neon-like Colors for Dark Theme Boxes
        color_palette = [
            '#00f5d4', '#ff00ff', '#39ff14', '#ffff00', '#00a8ff', '#ff5733', '#f8f8f8'
        ]
        color_map = {label: color for label, color in zip(sorted_labels, color_palette * (len(sorted_labels) // len(color_palette) + 1))}

        line_width = max(2, min(5, int(width * 0.006)))
        font_size = max(14, min(28, int(width * 0.035)))
        try:
            # Try common bold sans-serif fonts first
            font = ImageFont.truetype("arialbd.ttf", font_size) # Bold Arial (Windows)
        except IOError:
            try:
                font = ImageFont.truetype("Arial Bold.ttf", font_size) # Bold Arial (macOS/Other)
            except IOError:
                try:
                    font = ImageFont.truetype("arial.ttf", font_size) # Regular Arial fallback
                except IOError:
                    try:
                        font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size) # Common Linux bold font
                    except IOError:
                        print("Warning: Bold/Regular Arial/DejaVuSans fonts not found. Using default PIL font.")
                        font = ImageFont.load_default(size=font_size) # Use default PIL font with size attempt

        for box in valid_box_objects: # Iterate over validated/parsed objects
            y_min, x_min, y_max, x_max = box.box_2d
            label = box.label
            object_name = box.object_name # Use the attribute from the Pydantic model

            display_label = f"{object_name} ({label})" if object_name and object_name != "Unknown object" else label

            # Ensure coordinates are within image bounds after scaling
            abs_y_min = max(0, min(height, int(y_min / 1000 * height)))
            abs_x_min = max(0, min(width, int(x_min / 1000 * width)))
            abs_y_max = max(0, min(height, int(y_max / 1000 * height)))
            abs_x_max = max(0, min(width, int(x_max / 1000 * width)))

            # Skip drawing if box dimensions are invalid after clamping
            if abs_x_min >= abs_x_max or abs_y_min >= abs_y_max:
                print(f"Skipping drawing invalid box dimensions for {object_name}: [{abs_y_min}, {abs_x_min}, {abs_y_max}, {abs_x_max}]")
                continue

            color = color_map.get(label, '#f8f8f8') # Default to white/light gray

            # Draw Rectangle Outline
            draw.rectangle([(abs_x_min, abs_y_min), (abs_x_max, abs_y_max)], outline=color, width=line_width)

            # --- Draw Text Label with Background ---
            # Calculate text size accurately
            try:
                 # Use textbbox for potentially better accuracy with different fonts
                 text_bbox_calc = draw.textbbox((0, 0), display_label, font=font)
                 text_width = text_bbox_calc[2] - text_bbox_calc[0]
                 text_height = text_bbox_calc[3] - text_bbox_calc[1]
            except AttributeError: # Fallback for older PIL/Pillow or default font
                 text_width, text_height = draw.textlength(display_label, font=font), font_size # Approximate height

            text_x = abs_x_min + line_width
            text_y = abs_y_min + line_width

            bg_x0 = abs_x_min
            bg_y0 = abs_y_min
            # Add padding around text for the background
            bg_x1 = text_x + text_width + line_width
            bg_y1 = text_y + text_height + line_width

            # Ensure background doesn't exceed image bounds
            bg_x1 = min(bg_x1, width)
            bg_y1 = min(bg_y1, height)

            # Ensure background coordinates are valid
            if bg_x0 < bg_x1 and bg_y0 < bg_y1:
                 draw.rectangle((bg_x0, bg_y0, bg_x1, bg_y1), fill=color)
                 draw.text((text_x, text_y), display_label, fill='#1a1a2e', font=font) # Dark text for contrast
            else:
                 print(f"Skipping drawing text background for {object_name} due to invalid dimensions.")


        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG', quality=92) # Save as JPEG
        img_byte_arr = img_byte_arr.getvalue()
        return img_byte_arr
    except Exception as e:
        print(f"Error drawing boxes: {e}")
        traceback.print_exc() # Print stack trace for drawing errors
        return image_bytes # Return original on error

# --- Gemini API Interaction ---
def classify_waste(image_bytes, use_test_model=False):
    """
    Classifies waste items in an image using the Gemini API.

    Args:
        image_bytes: The image data as bytes.
        use_test_model: Boolean flag to use the Gemini 2.5 Flash test model.

    Returns:
        A tuple containing:
        - A list of validated BoundingBox objects (or None on critical error,
          or empty list if no objects found/validated).
        - A status message string.
    """
    response = None # Initialize response to None
    cleaned_response_text = "" # Initialize cleaned text
    try:
        # --- Select Model and Configuration ---
        if use_test_model:
            model_name = MODEL_TEST
            # Corrected structure for thinking budget
            # NOTE: Please verify this parameter name ('thinking_budget') with official SDK documentation if issues persist.
            generation_config = genai.types.GenerationConfig(
                response_mime_type='application/json',
                temperature=0.3,
                # Add thinking_budget directly here
                thinking_budget=1024
            )
            print(f"Using TEST model: {model_name} with thinking budget: 1024")
        else:
            model_name = MODEL_DEFAULT
            # Use GenerationConfig constructor here too for consistency
            generation_config = genai.types.GenerationConfig(
                 response_mime_type='application/json',
                 temperature=0.3
            )
            print(f"Using DEFAULT model: {model_name}")

        # --- Initialize Model and Prepare Request ---
        client = genai.GenerativeModel(model_name=model_name)
        image_part = {"mime_type": "image/jpeg", "data": image_bytes} # Assuming JPEG input
        # System prompt remains the same
        prompt = """
        You are Veridia Vision, a waste classification assistant.
        Analyze the image and identify distinct objects. For each object:
        1. Name the specific object (e.g., "plastic bottle", "banana peel", "cardboard box")
        2. Classify it as 'recyclable', 'non-recyclable', or 'organic'
        3. Classify it according to the rules of Waterloo Region.
        4. Classify humans as 'human'.
        5. Classify chips packets, chocolate wrappers, etc. as 'non-recyclable'.
        6. If a bottle has liquid or a container has food, classify as 'non-recyclable'.

        Return ONLY a valid JSON array of objects. Each object must have the following format:
        {
            "box_2d": [y_min, x_min, y_max, x_max],
            "label": "recyclable" or "non-recyclable" or "organic" or "human",
            "object_name": "name of the specific object"
        }

        The coordinates for "box_2d" MUST be normalized integers between 0 and 1000 (inclusive).
        The "label" MUST be one of: "recyclable", "non-recyclable", "organic", "human".
        If no relevant objects are found, return an empty JSON array: [].
        Do not include any explanatory text before or after the JSON array. Adhere strictly to the JSON format.
        """

        # --- Generate Content ---
        response = client.generate_content(
            contents=[prompt, image_part],
            generation_config=generation_config, # Pass the GenerationConfig object
            # Consider adding safety_settings if needed
            # safety_settings=[...]
        )

        # --- Process Response ---
        print(f"Raw Gemini Response Text Length: {len(response.text)}") # Debug length

        # Attempt to strip potential markdown/code blocks more robustly
        cleaned_response_text = response.text.strip()
        if cleaned_response_text.startswith("```json"):
             cleaned_response_text = cleaned_response_text[len("```json"):].strip()
        if cleaned_response_text.endswith("```"):
             cleaned_response_text = cleaned_response_text[:-len("```")].strip()

        if not cleaned_response_text:
            print("Gemini returned an empty response after stripping.")
            return [], "No classifiable objects detected (empty AI response)."

        # --- Parse and Validate JSON ---
        response_json = json.loads(cleaned_response_text)

        if not isinstance(response_json, list):
            print(f"Validation Error: Expected a list, got {type(response_json)}")
            # Try to return the raw JSON if it's not a list but might be useful
            return None, f"Error: AI response was not a list ({type(response_json).__name__})."


        # --- Validate individual items using Pydantic ---
        validated_boxes = []
        validation_errors = []
        for i, box_data in enumerate(response_json):
             try:
                 # Add default object_name if missing before validation
                 if 'object_name' not in box_data:
                      box_data['object_name'] = 'Unknown object'
                 validated_box = BoundingBox(**box_data)
                 # Additional checks
                 if not (isinstance(validated_box.box_2d, list) and len(validated_box.box_2d) == 4 and all(isinstance(c, int) and 0 <= c <= 1000 for c in validated_box.box_2d)):
                      raise ValueError("box_2d format invalid (must be list of 4 ints 0-1000)")
                 if validated_box.label not in ["recyclable", "non-recyclable", "organic", "human"]:
                      raise ValueError(f"Invalid label '{validated_box.label}'")

                 validated_boxes.append(validated_box)
             except (ValidationError, ValueError) as e:
                 print(f"Validation Error for item {i}: {e} - Data: {box_data}")
                 validation_errors.append(f"Item {i}: {str(e)[:100]}") # Collect errors

        if validation_errors:
             # Return successfully validated boxes along with error message
             status_suffix = f" (Validation errors: {'; '.join(validation_errors)})"
        else:
             status_suffix = ""


        # --- Generate Status Message ---
        num_boxes = len(validated_boxes)
        if num_boxes == 0:
            status = "No valid objects detected." + status_suffix
        elif num_boxes == 1:
            box = validated_boxes[0]
            object_name = box.object_name
            status = f"Detected 1 {box.label} item: {object_name}." + status_suffix
        else:
            counts = {}
            object_names = []
            for box in validated_boxes:
                counts[box.label] = counts.get(box.label, 0) + 1
                object_names.append(box.object_name)

            status_parts = [f"{count} {label}" for label, count in counts.items()]
            object_list = ", ".join(object_names)
            status = f"Detected {num_boxes} items: {', '.join(status_parts)}. Objects: {object_list}." + status_suffix

        return validated_boxes, status # Return potentially partial list and status

    # --- Exception Handling ---
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from Gemini response: {e}")
        print(f"Response text that failed: '{cleaned_response_text[:500]}...'")
        return None, f"Error: Could not parse AI response. {str(e)[:100]}"
    # Keep Pydantic validation error separate if needed, though handled above now
    # except ValidationError as e:
    #     print(f"Validation Error processing Gemini response: {e}")
    #     return None, f"Error: AI response format incorrect. {str(e)[:100]}"
    except Exception as e:
        print(f"Error during Gemini API call or processing: {e}")
        traceback.print_exc() # Print full stack trace for unexpected errors
        error_details = f"Type: {type(e).__name__}"
        # Check for specific Gemini API feedback if the 'response' object exists
        if response and hasattr(response, 'prompt_feedback'):
             try:
                  # Accessing prompt_feedback might raise an error itself if generation failed early
                  error_details += f". Prompt Feedback: {response.prompt_feedback}"
             except Exception as fb_e:
                  print(f"Could not access prompt_feedback: {fb_e}")
        elif response and hasattr(response, 'error'): # Check if response itself indicates an error
             error_details += f". Response Error Field: {response.error}"

        print(f"API Error Details: {error_details}")
        # Check if the error message itself indicates an unknown field
        if "Unknown field for GenerationConfig" in str(e):
             return None, f"API Config Error: Check parameter names (e.g., 'thinking_budget'). {str(e)[:100]}"
        else:
             return None, f"Error: AI communication failed. {str(e)[:100]}"


# --- Flask Routes ---
# ... (Routes remain the same as previous version) ...
@app.route('/')
def index():
    # Renders the main landing page (index.html)
    # This page's JS should NOT send the use_test_model flag
    print("Serving index.html for root route /") # Add log
    return render_template('index.html') # CHANGED TO INDEX.HTML

@app.route('/camera')
def camera_page():
    # Renders the main camera page (templates/camera.html)
    # This page's JS should NOT send the use_test_model flag
    print("Serving camera.html for route /camera") # Add log
    return render_template('camera.html')

@app.route('/camera_test')
def camera_test_page():
    # Renders the test page (templates/camera_test.html)
    # This page's JS WILL send the use_test_model flag
    print("Serving camera_test.html for route /camera_test") # Add log
    return render_template('camera_test.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    """Processes a single frame (image) from the frontend."""
    try:
        data = request.get_json()
        if not data or 'image_data' not in data:
            print("Error: Request missing image_data field.")
            return jsonify({"error": "Missing image_data", "status": "Client error: No image data received."}), 400

        image_data_url = data['image_data']
        # Determine if the test model should be used based on the flag from frontend
        use_test_model_flag = data.get('use_test_model', False) # Default to False
        print(f"Received process_frame request. use_test_model={use_test_model_flag}")


        # --- Decode Image ---
        try:
            # Split header (e.g., "data:image/jpeg;base64,") from encoded data
            header, encoded = image_data_url.split(",", 1)
            image_bytes = base64.b64decode(encoded)
            # Extract mime type (make robust)
            mime_type = 'image/jpeg' # Default assumption
            if ':' in header and ';' in header:
                 mime_part = header.split(':')[1].split(';')[0]
                 if '/' in mime_part: # Basic check for valid mime type format
                      mime_type = mime_part
            print(f"Decoded image with assumed mime type: {mime_type}")
            # Optional: Add stricter validation if needed, e.g., check allowed types
            # if mime_type not in ['image/jpeg', 'image/png', 'image/webp']:
            #     print(f"Warning: Received unexpected mime type: {mime_type}. Processing as JPEG.")
            #     mime_type = 'image/jpeg' # Standardize

        except (ValueError, base64.binascii.Error, IndexError) as e:
            print(f"Base64 decoding or header parsing error: {e}")
            return jsonify({"error": "Invalid image data format", "status": "Client error: Bad image format."}), 400


        # --- Processing with selected model ---
        start_time = time.time()
        # Pass the flag to the classification function
        bounding_boxes, status_message = classify_waste(image_bytes, use_test_model=use_test_model_flag)
        ai_time = time.time() - start_time
        print(f"Gemini classification took {ai_time:.2f} seconds. Status: {status_message}")


        # --- Draw Boxes and Prepare Response ---
        processed_image_bytes = image_bytes # Default to original
        boxes_found = False
        object_details_list = []

        if bounding_boxes is None:
            # Error occurred in classify_waste, status_message has details
            print(f"classify_waste returned None. Status: {status_message}")
        elif not bounding_boxes:
            # No objects found or validated
            print("classify_waste returned empty or invalid list. No objects detected/drawn.")
        else:
            # Attempt to draw boxes only if we got valid data
            boxes_found = True
            start_draw_time = time.time()
            drawn_bytes = draw_bounding_boxes(image_bytes, bounding_boxes)
            draw_time = time.time() - start_draw_time
            print(f"Drawing boxes took {draw_time:.2f} seconds.")

            if drawn_bytes is not None and drawn_bytes != image_bytes:
                processed_image_bytes = drawn_bytes
            elif drawn_bytes is None: # Check if drawing explicitly failed
                print("Drawing boxes failed, returning original image.")
                status_message += " (Error drawing boxes)"
            # else: drawing might have returned original image due to no valid boxes

            # Prepare details list from the (potentially filtered) bounding_boxes
            object_details_list = [
                 {"name": box.object_name, "classification": box.label}
                 for box in bounding_boxes # Use the list returned by classify_waste
            ]


        # --- Prepare Response JSON ---
        processed_image_base64 = base64.b64encode(processed_image_bytes).decode('utf-8')
        # Use the determined mime_type for the result data URL
        result_image_data_url = f"data:{mime_type};base64,{processed_image_base64}"

        return jsonify({
            "status": status_message,
            "processed_image_data": result_image_data_url,
            "boxes_found": boxes_found, # Indicates if boxes were *attempted* to be drawn
            "object_details": object_details_list # List of detected objects
        })

    except Exception as e:
        print(f"Critical Error in /process_frame: {e}")
        traceback.print_exc() # Print full stack trace
        return jsonify({"error": "An internal server error occurred", "status": "Server error during processing."}), 500


# --- Run ---
if __name__ == '__main__':
    # Use waitress or gunicorn for production instead of Flask's debug server
    # For development:
    # Set debug=False if deploying, True for local development only
    # Host 0.0.0.0 makes it accessible on your network, use 127.0.0.1 for local only
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))

