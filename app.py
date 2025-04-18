# Complete app.py with separate logic for /camera_test

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
    object_name: str = "Unknown object"

# --- Image Processing ---
# (draw_bounding_boxes function remains the same as in your provided code)
def draw_bounding_boxes(image_bytes, bounding_boxes):
    """Draws bounding boxes with a theme suitable for dark UI."""
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        width, height = img.size
        draw = ImageDraw.Draw(img)
        if not all(isinstance(box, BoundingBox) for box in bounding_boxes):
             print("Warning: draw_bounding_boxes received invalid format.")
             valid_boxes = [box for box in bounding_boxes if isinstance(box, BoundingBox)]
             if not valid_boxes: return image_bytes
             bounding_boxes = valid_boxes
        labels = sorted(list(set(box.label for box in bounding_boxes)))
        color_palette = ['#00f5d4', '#ff00ff', '#39ff14', '#ffff00', '#00a8ff', '#ff5733', '#f8f8f8']
        color_map = {label: color for label, color in zip(labels, color_palette * (len(labels) // len(color_palette) + 1))}
        line_width = max(2, min(5, int(width * 0.006)))
        font_size = max(14, min(28, int(width * 0.035)))
        try: font = ImageFont.truetype("arialbd.ttf", font_size)
        except IOError:
            try: font = ImageFont.truetype("Arial Bold.ttf", font_size)
            except IOError:
                try: font = ImageFont.truetype("arial.ttf", font_size)
                except IOError:
                    try: font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
                    except IOError: font = ImageFont.load_default()
        for box in bounding_boxes:
            if not hasattr(box, 'box_2d') or len(box.box_2d) != 4: continue
            try:
                y_min, x_min, y_max, x_max = box.box_2d
                abs_y_min = int(y_min / 1000 * height); abs_x_min = int(x_min / 1000 * width)
                abs_y_max = int(y_max / 1000 * height); abs_x_max = int(x_max / 1000 * width)
            except (TypeError, ValueError) as coord_err: continue
            label = getattr(box, 'label', 'unknown'); object_name = getattr(box, 'object_name', 'Object')
            display_label = f"{object_name} ({label})" if object_name != "Unknown object" else label
            color = color_map.get(label, '#f8f8f8')
            draw.rectangle([(abs_x_min, abs_y_min), (abs_x_max, abs_y_max)], outline=color, width=line_width)
            text_position = (abs_x_min + line_width, abs_y_min + line_width)
            try:
                 text_bbox = draw.textbbox(text_position, display_label, font=font, spacing=4)
                 bg_x1 = min(text_bbox[2] + line_width * 2, width); bg_y1 = min(text_bbox[3] + line_width, height)
                 draw.rectangle((abs_x_min, abs_y_min, bg_x1, bg_y1), fill=color)
                 draw.text(text_position, display_label, fill='#1a1a2e', font=font)
            except Exception as text_err: print(f"Warn: Text draw error: {text_err}")
        img_byte_arr = io.BytesIO(); img.save(img_byte_arr, format='JPEG', quality=92)
        return img_byte_arr.getvalue()
    except Exception as e: print(f"Error in draw_bounding_boxes: {e}"); return image_bytes

# --- Gemini API Interaction (Original for /camera) ---
def classify_waste_original(image_bytes):
    """Classifies waste using Gemini 2.0 Flash (Original Model)."""
    model_name = "gemini-2.0-flash" # <<< USES 2.0 FLASH
    print(f"Calling ORIGINAL classify_waste with model: {model_name}")
    try:
        client = genai.GenerativeModel(model_name=model_name)
        image_part = {"mime_type": "image/jpeg", "data": image_bytes}
        # Using the prompt specified in your app.py
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
        cleaned_response_text = response.text.strip().strip('```json').strip('```').strip()
        if not cleaned_response_text: return [], "No objects detected."
        response_json = json.loads(cleaned_response_text)
        if not isinstance(response_json, list): return None, "Error: AI response not list."
        validated_boxes = [BoundingBox(**box) for box in response_json]
        # (Status message generation logic - same as before)
        num = len(validated_boxes); status = "No objects detected."
        if num == 1: box = validated_boxes[0]; status = f"Detected 1 {box.label} item: {box.object_name}."
        elif num > 1: counts = {}; names = []; [ (counts.update({b.label: counts.get(b.label, 0) + 1}), names.append(b.object_name)) for b in validated_boxes ]; parts = [f"{c} {l}" for l, c in counts.items()]; status = f"Detected {num} items: {'; '.join(parts)}. Objects: {', '.join(names)}."
        return validated_boxes, status
    except Exception as e:
        print(f"Error in classify_waste_original ({model_name}): {e}")
        return None, f"Error: AI communication failed ({model_name})."

# --- Gemini API Interaction (NEW for /camera_test) ---
def classify_waste_test(image_bytes):
    """Classifies waste using Gemini 2.5 Flash Preview (Test Model)."""
    model_name = "gemini-2.5-flash-preview-04-17" # <<< USES 2.5 FLASH PREVIEW
    print(f"Calling TEST classify_waste with model: {model_name}")
    try:
        client = genai.GenerativeModel(model_name=model_name)
        image_part = {"mime_type": "image/jpeg", "data": image_bytes}
        # Using the same detailed prompt asking for classification
        prompt = """
        You are Veridia Vision, a waste classification assistant for Waterloo Region, Ontario, Canada.
        Analyze the image and identify distinct objects. For each object:
        1. Name the specific object (e.g., "plastic bottle", "banana peel", "cardboard box", "human").
        2. Classify it STRICTLY as 'recyclable', 'non-recyclable', or 'organic' according to Waterloo Region rules.
           - Clean plastic bottles/containers (#1, #2, #5), glass bottles/jars, metal cans, paper, cardboard are 'recyclable'.
           - Food waste, coffee grounds, soiled paper towels/napkins are 'organic'.
           - Plastic bags/film, styrofoam, coffee cups, chip bags, wrappers, ceramics, items with food/liquid residue are 'non-recyclable'.
           - Identify humans as 'human' and classify them as 'non-recyclable' for waste purposes.
        3. Provide the bounding box.

        Return ONLY a valid JSON array of objects. Each object MUST have the following format:
        {
            "box_2d": [y_min, x_min, y_max, x_max],
            "label": "recyclable" or "non-recyclable" or "organic" or "human",
            "object_name": "name of the specific object"
        }
        The coordinates for "box_2d" MUST be normalized integers between 0 and 1000 (inclusive).
        If no relevant objects are found, return an empty JSON array: [].
        Do not include any explanatory text before or after the JSON array.
        """
        # No thinking_budget specified for simplicity/stability for now
        response = client.generate_content(
            contents=[prompt, image_part],
            generation_config={'response_mime_type': 'application/json'},
        )
        cleaned_response_text = response.text.strip().strip('```json').strip('```').strip()
        if not cleaned_response_text: return [], "No objects detected."
        response_json = json.loads(cleaned_response_text)
        if not isinstance(response_json, list): return None, "Error: AI response not list."
        validated_boxes = [BoundingBox(**box) for box in response_json]
        # (Status message generation logic - same as before)
        num = len(validated_boxes); status = "No objects detected."
        if num == 1: box = validated_boxes[0]; status = f"Detected 1 {box.label} item: {box.object_name} (2.5 Flash)." # Added model marker
        elif num > 1: counts = {}; names = []; [ (counts.update({b.label: counts.get(b.label, 0) + 1}), names.append(b.object_name)) for b in validated_boxes ]; parts = [f"{c} {l}" for l, c in counts.items()]; status = f"Detected {num} items: {'; '.join(parts)}. Objects: {', '.join(names)} (2.5 Flash)." # Added model marker
        return validated_boxes, status
    except Exception as e:
        print(f"Error in classify_waste_test ({model_name}): {e}")
        return None, f"Error: AI communication failed ({model_name})."


# --- Flask Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/camera')
def camera_page():
    return render_template('camera.html') # Uses original camera template

@app.route('/camera_test')
def camera_test_page():
    return render_template('camera_test.html') # Uses the new test template

@app.route('/changelog')
def changelog_page():
    return render_template('changelog.html')

# --- Processing Endpoint (Original for /camera) ---
@app.route('/process_frame', methods=['POST'])
def process_frame():
    """Processes frame using the ORIGINAL model (Gemini 2.0 Flash)."""
    print("--- Received request on /process_frame (using Gemini 2.0) ---")
    try:
        data = request.get_json()
        if not data or 'image_data' not in data: return jsonify({"error": "Missing image_data"}), 400
        image_data_url = data['image_data']
        try: header, encoded = image_data_url.split(",", 1); image_bytes = base64.b64decode(encoded); mime_type = header.split(':')[1].split(';')[0]; mime_type = mime_type if mime_type in ['image/jpeg', 'image/png', 'image/webp'] else 'image/jpeg'
        except Exception as e: print(f"Img parse err: {e}"); return jsonify({"error": "Invalid image"}), 400

        start_time = time.time()
        # *** CALLS ORIGINAL FUNCTION ***
        bounding_boxes, status_message = classify_waste_original(image_bytes)
        ai_time = time.time() - start_time
        print(f"Original model processing took {ai_time:.2f}s. Status: {status_message}")

        processed_image_bytes = image_bytes; boxes_found = False; object_details_list = []
        if bounding_boxes: # Check if list is not None and not empty
            boxes_found = True
            drawn_bytes = draw_bounding_boxes(image_bytes, bounding_boxes) # Draw boxes for original
            if drawn_bytes != image_bytes: processed_image_bytes = drawn_bytes
            object_details_list = [{"name": getattr(b, 'object_name', 'N/A'), "classification": getattr(b, 'label', 'N/A')} for b in bounding_boxes]
        elif bounding_boxes is None: print(f"Error status from original classify: {status_message}") # Error case

        processed_image_base64 = base64.b64encode(processed_image_bytes).decode('utf-8')
        result_image_data_url = f"data:{mime_type};base64,{processed_image_base64}"
        return jsonify({"status": status_message, "processed_image_data": result_image_data_url, "boxes_found": boxes_found, "object_details": object_details_list})
    except Exception as e: print(f"Error in /process_frame: {e}"); import traceback; traceback.print_exc(); return jsonify({"error": "Server error"}), 500

# --- NEW Processing Endpoint (for /camera_test) ---
@app.route('/process_frame_test', methods=['POST'])
def process_frame_test():
    """Processes frame using the TEST model (Gemini 2.5 Flash Preview)."""
    print("--- Received request on /process_frame_test (using Gemini 2.5) ---")
    try:
        data = request.get_json()
        if not data or 'image_data' not in data: return jsonify({"error": "Missing image_data"}), 400
        image_data_url = data['image_data']
        try: header, encoded = image_data_url.split(",", 1); image_bytes = base64.b64decode(encoded); mime_type = header.split(':')[1].split(';')[0]; mime_type = mime_type if mime_type in ['image/jpeg', 'image/png', 'image/webp'] else 'image/jpeg'
        except Exception as e: print(f"Img parse err: {e}"); return jsonify({"error": "Invalid image"}), 400

        start_time = time.time()
        # *** CALLS TEST FUNCTION ***
        bounding_boxes, status_message = classify_waste_test(image_bytes)
        ai_time = time.time() - start_time
        print(f"Test model processing took {ai_time:.2f}s. Status: {status_message}")

        processed_image_bytes = image_bytes; boxes_found = False; object_details_list = []
        if bounding_boxes: # Check if list is not None and not empty
            boxes_found = True
            # Decide if you want boxes drawn for the test page too
            # drawn_bytes = draw_bounding_boxes(image_bytes, bounding_boxes)
            # if drawn_bytes != image_bytes: processed_image_bytes = drawn_bytes
            object_details_list = [{"name": getattr(b, 'object_name', 'N/A'), "classification": getattr(b, 'label', 'N/A')} for b in bounding_boxes]
        elif bounding_boxes is None: print(f"Error status from test classify: {status_message}") # Error case

        processed_image_base64 = base64.b64encode(processed_image_bytes).decode('utf-8')
        result_image_data_url = f"data:{mime_type};base64,{processed_image_base64}" # Test page might ignore this if not drawing boxes
        return jsonify({"status": status_message, "processed_image_data": result_image_data_url, "boxes_found": boxes_found, "object_details": object_details_list})
    except Exception as e: print(f"Error in /process_frame_test: {e}"); import traceback; traceback.print_exc(); return jsonify({"error": "Server error"}), 500


# --- Run ---
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
```

**Explanation of Changes in `app.py`:**

1.  **`classify_waste_original` function:** This is essentially your previous `classify_waste` function, hardcoded to use `"gemini-2.0-flash"`.
2.  **`classify_waste_test` function:** A new function, almost identical to the original, but hardcoded to use `"gemini-2.5-flash-preview-04-17"`. I added `(2.5 Flash)` to the status message it generates just for clarity during testing.
3.  **`/process_frame` endpoint:** This *remains unchanged* and continues to call `classify_waste_original` (using 2.0 Flash).
4.  **`/process_frame_test` endpoint:** A *new* endpoint that calls the new `classify_waste_test` function (using 2.5 Flash).

**Crucial Next Step: Modify `camera_test.html`**

You **must** now edit the JavaScript inside your `templates/camera_test.html` file. Find the line that makes the `fetch` call (it's inside the `captureAndProcessFrame` function) and change the URL:

* **Find:** `const response = await fetch('/process_frame', { ... });`
* **Change to:** `const response = await fetch('/process_frame_test', { ... });`

This change ensures that when the test page sends an image, it hits the new backend endpoint which uses the Gemini 2.5 Flash model. Your original `camera.html` will keep calling `/process_frame` and use the 2.0 Flash mod
