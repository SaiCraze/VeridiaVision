import os
import io
import base64
import json
import time
from flask import Flask, render_template, request, jsonify, redirect, url_for
from PIL import Image, ImageDraw, ImageFont
import google.generativeai as genai
from pydantic import BaseModel, ValidationError
from dotenv import load_dotenv
import numpy as np

# --- Configuration & Setup ---
load_dotenv()
app = Flask(__name__)
app.secret_key = os.urandom(24)

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable not set.")
genai.configure(api_key=api_key)

# --- Data Model ---
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
        labels = sorted(list(set(box.label for box in bounding_boxes)))

        # Vibrant Neon-like Colors for Dark Theme Boxes
        color_palette = [
            '#00f5d4', '#ff00ff', '#39ff14', '#ffff00', '#00a8ff', '#ff5733', '#f8f8f8'
        ]
        color_map = {label: color for label, color in zip(labels, color_palette * (len(labels) // len(color_palette) + 1))}

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
                        font = ImageFont.load_default() # Absolute fallback

        for box in bounding_boxes:
            y_min, x_min, y_max, x_max = box.box_2d
            label = box.label
            object_name = getattr(box, 'object_name', 'Object')

            display_label = f"{object_name} ({label})" if object_name and object_name != "Unknown object" else label

            abs_y_min = int(y_min / 1000 * height)
            abs_x_min = int(x_min / 1000 * width)
            abs_y_max = int(y_max / 1000 * height)
            abs_x_max = int(x_max / 1000 * width)
            color = color_map.get(label, '#f8f8f8')

            # Draw Rectangle Outline
            draw.rectangle([(abs_x_min, abs_y_min), (abs_x_max, abs_y_max)], outline=color, width=line_width)

            # --- Draw Text Label with Background ---
            text_position = (abs_x_min + line_width, abs_y_min + line_width)
            text_bbox = draw.textbbox(text_position, display_label, font=font, spacing=4)

            bg_x0 = abs_x_min
            bg_y0 = abs_y_min
            bg_x1 = text_bbox[2] + line_width * 2
            bg_y1 = text_bbox[3] + line_width

            bg_x1 = min(bg_x1, width)
            bg_y1 = min(bg_y1, height)

            draw.rectangle((bg_x0, bg_y0, bg_x1, bg_y1), fill=color)
            draw.text(text_position, display_label, fill='#1a1a2e', font=font) # Dark text

        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG', quality=92)
        img_byte_arr = img_byte_arr.getvalue()
        return img_byte_arr
    except Exception as e:
        print(f"Error drawing boxes: {e}")
        return image_bytes # Return original on error

# --- Gemini API Interaction ---
def classify_waste(image_bytes):
    try:
        client = genai.GenerativeModel(model_name="gemini-2.0-flash")
        image_part = {"mime_type": "image/jpeg", "data": image_bytes}
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
            "label": "recyclable" or "non-recyclable" or "organic",
            "object_name": "name of the specific object"
        }

        The coordinates for "box_2d" MUST be normalized integers between 0 and 1000 (inclusive).
        If no relevant objects are found, return an empty JSON array: [].
        Do not include any explanatory text before or after the JSON array.
        """
        response = client.generate_content(
            contents=[prompt, image_part],
            generation_config={'response_mime_type': 'application/json'},
        )
        print(f"Raw Gemini Response Text Length: {len(response.text)}") # Debug length instead of full text if too long

        # Attempt to strip potential markdown/code blocks
        cleaned_response_text = response.text.strip().strip('```json').strip('```').strip()
        if not cleaned_response_text:
             print("Gemini returned an empty response after stripping.")
             return [], "No classifiable objects detected (empty AI response)." # Return empty list and status

        response_json = json.loads(cleaned_response_text)

        # Validate structure - ensure it's a list
        if not isinstance(response_json, list):
            print(f"Validation Error: Expected a list, got {type(response_json)}")
            return None, f"Error: AI response was not a list."

        validated_boxes = [BoundingBox(**box) for box in response_json]

        num_boxes = len(validated_boxes)
        if num_boxes == 0:
            status = "No classifiable objects detected."
        elif num_boxes == 1:
            box = validated_boxes[0]
            object_name = getattr(box, 'object_name', 'item')
            status = f"Detected 1 {box.label} item: {object_name}."
        else:
            counts = {}
            object_names = []
            for box in validated_boxes:
                counts[box.label] = counts.get(box.label, 0) + 1
                object_names.append(getattr(box, 'object_name', 'item'))

            status_parts = [f"{count} {label}" for label, count in counts.items()]
            object_list = ", ".join(object_names)
            status = f"Detected {num_boxes} items: {', '.join(status_parts)}. Objects: {object_list}."

        return validated_boxes, status

    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from Gemini response: {e}")
        print(f"Response text that failed: '{cleaned_response_text[:500]}...'") # Log beginning of failing text
        return None, f"Error: Could not parse AI response. {str(e)[:100]}"
    except ValidationError as e:
        print(f"Validation Error processing Gemini response: {e}")
        return None, f"Error: AI response format incorrect. {str(e)[:100]}"
    except Exception as e:
        print(f"Error during Gemini API call or processing: {e}")
        # Attempt to get more specific feedback if available
        error_details = "No specific details available."
        if hasattr(response, 'prompt_feedback'):
             error_details = f"Prompt Feedback: {response.prompt_feedback}"
        elif hasattr(response, 'error'):
             error_details = f"Response Error Field: {response.error}"
        print(f"API Error Details: {error_details}")
        return None, f"Error: AI communication failed. {str(e)[:100]}"


# --- Flask Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/camera')
def camera_page():
    return render_template('camera.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    try:
        data = request.get_json()
        if not data or 'image_data' not in data:
            return jsonify({"error": "Missing image_data", "status": "Client error: No image data received."}), 400

        image_data_url = data['image_data']
        try:
            header, encoded = image_data_url.split(",", 1)
            image_bytes = base64.b64decode(encoded)
            mime_type = header.split(':')[1].split(';')[0]
            if mime_type not in ['image/jpeg', 'image/png', 'image/webp']:
                 print(f"Warning: Received unexpected mime type: {mime_type}. Processing as JPEG.")
                 mime_type = 'image/jpeg' # Standardize for processing
        except (ValueError, base64.binascii.Error, IndexError) as e:
             print(f"Base64 decoding or header parsing error: {e}")
             return jsonify({"error": "Invalid image data format", "status": "Client error: Bad image format."}), 400

        # --- Processing ---
        start_time = time.time()
        bounding_boxes, status_message = classify_waste(image_bytes)
        ai_time = time.time() - start_time
        print(f"Gemini classification took {ai_time:.2f} seconds.")

        processed_image_bytes = image_bytes # Default to original
        boxes_found = False
        object_details_list = []

        if bounding_boxes is None:
            # Error already captured in status_message by classify_waste
            print(f"classify_waste returned None. Status: {status_message}")
            pass # status_message already contains the error description
        elif not bounding_boxes:
            # No objects found, status_message reflects this
            print("classify_waste returned empty list. No objects detected.")
            pass
        else:
            boxes_found = True
            start_draw_time = time.time()
            drawn_bytes = draw_bounding_boxes(image_bytes, bounding_boxes)
            draw_time = time.time() - start_draw_time
            print(f"Drawing boxes took {draw_time:.2f} seconds.")

            if drawn_bytes is not None and drawn_bytes != image_bytes:
                 processed_image_bytes = drawn_bytes
            elif drawn_bytes is None:
                 print("Drawing boxes failed, returning original image.")
                 status_message += " (Error drawing boxes)"

            object_details_list = [
                {"name": getattr(box, 'object_name', 'Unknown'), "classification": box.label}
                for box in bounding_boxes
            ]

        # --- Prepare Response ---
        processed_image_base64 = base64.b64encode(processed_image_bytes).decode('utf-8')
        # Use the potentially corrected mime_type here
        result_image_data_url = f"data:{mime_type};base64,{processed_image_base64}"

        return jsonify({
            "status": status_message,
            "processed_image_data": result_image_data_url,
            "boxes_found": boxes_found,
            "object_details": object_details_list
        })

    except Exception as e:
        print(f"Critical Error in /process_frame: {e}")
        import traceback
        traceback.print_exc() # Print full stack trace for debugging server errors
        return jsonify({"error": "An internal server error occurred", "status": "Server error during processing."}), 500


# --- Run ---
if __name__ == '__main__':
    # Use waitresses or gunicorn for production instead of Flask's debug server
    # For development:
    app.run(debug=True, host='0.0.0.0', port=5000)
