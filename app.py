import os
import io
import base64
import json
import time
from flask import Flask, render_template, request, jsonify # Removed redirect, url_for as not used in this file
from PIL import Image, ImageDraw, ImageFont
import google.generativeai as genai
from pydantic import BaseModel, Field, ValidationError # Added Field for clarity
from dotenv import load_dotenv
import traceback # For better error logging

# --- Configuration & Setup ---
load_dotenv()
app = Flask(__name__)
app.secret_key = os.urandom(24) # Good practice

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable not set.")
genai.configure(api_key=api_key)

# --- Data Model ---
class BoundingBox(BaseModel):
    # Coordinates must be integers between 0 and 1000
    box_2d: list[int] = Field(..., min_items=4, max_items=4)
    # Label must be one of the specified values
    label: str # Will be validated later against allowed labels
    object_name: str = "Unknown object" # Specific name of the item

    # Add custom validation for coordinates if Pydantic version supports root validators easily
    # or validate in the parsing loop.

# Allowed classification labels + Human
ALLOWED_LABELS = {'recyclable', 'non-recyclable', 'organic', 'Human'}

# --- Image Processing ---
def draw_bounding_boxes(image_bytes, bounding_boxes):
    """Draws bounding boxes with specific colors for different labels."""
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        width, height = img.size
        draw = ImageDraw.Draw(img)

        # Define colors for each specific label, including Human
        color_map = {
            'recyclable': '#00f5d4',      # Teal/Cyan
            'non-recyclable': '#ff4d4d', # Red
            'organic': '#39ff14',         # Lime Green
            'Human': '#f8f8f8',           # White/Light Gray
            'default': '#ffff00'          # Yellow for any unexpected label
        }

        # Dynamic sizing based on image dimensions
        line_width = max(2, min(5, int(width * 0.005)))
        font_size = max(14, min(24, int(width * 0.03)))
        try:
            # Prioritize common bold fonts
            font = ImageFont.truetype("arialbd.ttf", font_size)
        except IOError:
            try:
                font = ImageFont.truetype("Arial Bold.ttf", font_size)
            except IOError:
                try:
                    font = ImageFont.truetype("arial.ttf", font_size)
                except IOError:
                     try:
                         font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
                     except IOError:
                         try:
                              font = ImageFont.truetype("DejaVuSans.ttf", font_size)
                         except IOError:
                              print("Warning: Preferred fonts not found. Using default PIL font.")
                              font = ImageFont.load_default()

        for box in bounding_boxes:
            if not (len(box.box_2d) == 4 and all(isinstance(coord, int) for coord in box.box_2d)):
                 print(f"Warning: Skipping invalid box_2d format: {box.box_2d}")
                 continue
            y_min, x_min, y_max, x_max = box.box_2d
            if not all(0 <= coord <= 1000 for coord in [y_min, x_min, y_max, x_max]):
                 print(f"Warning: Skipping box with out-of-range coordinates (0-1000): {box.box_2d}")
                 continue
            if y_min >= y_max or x_min >= x_max:
                 print(f"Warning: Skipping box with invalid min/max coords: {box.box_2d}")
                 continue

            label = box.label
            object_name = getattr(box, 'object_name', 'Object')
            color = color_map.get(label, color_map['default']) # Use default color if label is unexpected

            # Determine display text: Use specific name for waste, just "Human" for people
            display_text = "Human" if label == "Human" else f"{object_name.title()} ({label.title()})"

            abs_y_min = int(y_min / 1000 * height)
            abs_x_min = int(x_min / 1000 * width)
            abs_y_max = int(y_max / 1000 * height)
            abs_x_max = int(x_max / 1000 * width)

            # Draw Rectangle Outline
            draw.rectangle([(abs_x_min, abs_y_min), (abs_x_max, abs_y_max)], outline=color, width=line_width)

            # --- Draw Text Label with Background ---
            text_x_pos = abs_x_min + line_width + 2
            text_y_pos = abs_y_min + line_width
            try:
                text_bbox = draw.textbbox((text_x_pos, text_y_pos), display_text, font=font)
            except AttributeError: # Fallback for older PIL
                 text_width, text_height = draw.textsize(display_text, font=font)
                 text_bbox = (text_x_pos, text_y_pos, text_x_pos + text_width, text_y_pos + text_height)

            bg_x0 = abs_x_min
            bg_y0 = abs_y_min
            bg_x1 = text_bbox[2] + line_width + 4
            bg_y1 = text_bbox[3] + line_width + 2
            bg_x1 = min(bg_x1, abs_x_max, width)
            bg_y1 = min(bg_y1, abs_y_max, height)
            bg_x0 = max(bg_x0, 0)
            bg_y0 = max(bg_y0, 0)

            if bg_x1 > bg_x0 and bg_y1 > bg_y0:
                draw.rectangle((bg_x0, bg_y0, bg_x1, bg_y1), fill=color)
                # Determine text color based on background brightness (simple heuristic)
                text_color = '#111111' # Dark text default
                # If background color is dark (e.g., potentially a dark custom color), use light text
                # This is a basic check, real check involves calculating luminance
                # if color.startswith('#') and int(color[1:3], 16) + int(color[3:5], 16) + int(color[5:7], 16) < 382: # ~50% brightness
                #    text_color = '#FFFFFF' # Light text for dark backgrounds
                if label == 'non-recyclable': # Ensure light text on red bg
                    text_color = '#FFFFFF'

                draw.text((text_x_pos, text_y_pos), display_text, fill=text_color, font=font)

        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG', quality=90)
        img_byte_arr = img_byte_arr.getvalue()
        return img_byte_arr
    except Exception as e:
        print(f"Error drawing bounding boxes: {e}")
        traceback.print_exc()
        return image_bytes # Return original on error

# --- Gemini API Interaction ---
def classify_waste(image_bytes):
    """Sends image to Gemini for classification based on specific rules."""
    try:
        client = genai.GenerativeModel(model_name="gemini-1.5-flash-latest")
        image_part = {"mime_type": "image/jpeg", "data": image_bytes}

        # --- ***** THE CRITICAL PROMPT ENGINEERING ***** ---
        prompt = f"""
        You are Veridia Vision, a waste classification assistant applying **Waterloo Region, Ontario, Canada** recycling and waste sorting rules.
        Analyze the provided image to identify distinct objects and people. For each identified item:
        1.  Provide its bounding box coordinates [y_min, x_min, y_max, x_max] as integers normalized between 0 and 1000.
        2.  Determine the specific `object_name` (e.g., "plastic water bottle", "banana peel", "person", "foil chip bag").
        3.  Assign a `label` based *strictly* on the following Waterloo Region rules:
            *   **Recyclable:** Items like empty plastic bottles & jugs (#1, #2, #5), empty glass bottles & jars, empty metal food & drink cans, paper (newspaper, flyers, magazines), cardboard boxes (flattened), empty cartons (milk, juice).
            *   **Organic:** Food scraps (fruit/veg peels, meat, bones, dairy), soiled paper products (paper towels, napkins, tissues, pizza boxes with food residue).
            *   **Non-recyclable:** Items like plastic bags/film, styrofoam, foil-lined chip bags, chocolate bar wrappers, coffee pods, broken glass/ceramics, textiles, wood, **any bottle or container with significant liquid/food residue remaining**.
            *   **Human:** If a person is clearly visible.

        **Specific Handling Rules:**
        *   **Liquids in Containers:** If a bottle, jar, or container clearly contains significant liquid or food residue, label it 'non-recyclable', regardless of the material. If it appears empty or nearly empty, classify the container material itself (e.g., plastic bottle as 'recyclable').
        *   **Wrappers:** Foil-lined chip bags, granola bar wrappers, and chocolate bar wrappers are 'non-recyclable'.
        *   **Human Presence:** If a person is visible, identify them with the label 'Human' and `object_name` "person".
        *   **Human Presenting Object:** If a person is holding or clearly presenting an object, identify BOTH the person (label: 'Human') AND the object separately with its correct waste classification (label: 'recyclable'/'non-recyclable'/'organic'). Focus the classification on the OBJECT being presented. Do not classify the human as waste.

        **Output Format:**
        Return ONLY a valid JSON array containing objects found. Each object MUST follow this exact structure:
        {{
            "box_2d": [y_min, x_min, y_max, x_max],
            "label": "{' | '.join(ALLOWED_LABELS)}",
            "object_name": "Specific Object or Person Name"
        }}

        Example:
        [
            {{"box_2d": [150, 200, 750, 800], "label": "recyclable", "object_name": "empty plastic water bottle"}},
            {{"box_2d": [50, 50, 950, 400], "label": "Human", "object_name": "person"}},
            {{"box_2d": [400, 600, 600, 900], "label": "non-recyclable", "object_name": "foil chip bag"}}
        ]

        If no relevant objects or people are detected, return an empty JSON array: [].
        Do NOT include markdown formatting (like ```json) or any text outside the JSON array.
        Ensure all coordinates are integers between 0 and 1000.
        """
        # --- ***** END OF PROMPT ***** ---


        generation_config = genai.types.GenerationConfig(
            response_mime_type="application/json",
            temperature=0.2 # Lower temperature for more deterministic adherence to rules
        )

        response = client.generate_content(
            contents=[prompt, image_part],
            generation_config=generation_config,
        )

        print(f"Raw Gemini Response Text Length: {len(response.text)}") # Debugging

        # Strict JSON parsing
        try:
            response_json = json.loads(response.text)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from Gemini response: {e}")
            print(f"Response text that failed: '{response.text[:500]}...'")
            # Try cleaning common issues only if direct parsing fails
            cleaned_response_text = response.text.strip().strip('```json').strip('```').strip()
            try:
                response_json = json.loads(cleaned_response_text)
                print("Successfully parsed JSON after cleaning.")
            except json.JSONDecodeError as inner_e:
                 print(f"Still failed to parse JSON after cleaning: {inner_e}")
                 return None, f"Error: Could not parse AI response. Invalid JSON received."

        if not isinstance(response_json, list):
            print(f"Validation Error: Expected a list, got {type(response_json)}")
            return None, "Error: AI response was not a valid JSON list."

        # Validate individual items using Pydantic and allowed labels
        validated_boxes = []
        validation_errors = []
        for i, box_data in enumerate(response_json):
            try:
                # Basic structure validation first
                if not isinstance(box_data, dict):
                     raise ValueError(f"Item {i} is not a dictionary.")
                if "label" not in box_data or "box_2d" not in box_data:
                     raise ValueError(f"Item {i} missing required keys ('label', 'box_2d').")

                # Validate label against allowed set
                if box_data.get("label") not in ALLOWED_LABELS:
                    raise ValueError(f"Item {i} has invalid label '{box_data.get('label')}'. Must be one of: {ALLOWED_LABELS}")

                # Validate with Pydantic model
                validated_box = BoundingBox(**box_data)

                # Explicit coordinate validation (redundant if Pydantic handles ranges well, but safe)
                y_min, x_min, y_max, x_max = validated_box.box_2d
                if not all(isinstance(coord, int) and 0 <= coord <= 1000 for coord in [y_min, x_min, y_max, x_max]):
                     raise ValueError(f"Item {i} coordinates out of range/type: {validated_box.box_2d}")
                if y_min >= y_max or x_min >= x_max:
                     raise ValueError(f"Item {i} invalid min/max coordinates: {validated_box.box_2d}")

                validated_boxes.append(validated_box)
            except (ValidationError, ValueError) as e:
                validation_errors.append(f"Item {i} ({box_data.get('object_name', 'Unknown')}): {e}")


        if validation_errors:
            print(f"Validation Errors in Gemini response items:\n" + "\n".join(validation_errors))
            # Decide: Fail entirely or proceed with valid boxes? Proceeding is often better UX.
            if not validated_boxes:
                 return None, f"Error: AI response format incorrect. No valid items found."

        # --- Generate Status Message (Handle 'Human' label) ---
        num_total_items = len(validated_boxes)
        waste_items = [box for box in validated_boxes if box.label != 'Human']
        human_items = [box for box in validated_boxes if box.label == 'Human']
        num_waste_items = len(waste_items)
        num_humans = len(human_items)

        status_parts = []
        if num_waste_items > 0:
            counts = {}
            object_names = []
            for box in waste_items:
                label = box.label
                counts[label] = counts.get(label, 0) + 1
                object_names.append(getattr(box, 'object_name', 'item').title())

            waste_summary_parts = []
            for label, count in counts.items():
                 plural = "s" if count > 1 else ""
                 waste_summary_parts.append(f"{count} {label.title()}{plural}")

            object_list_str = ", ".join(object_names)
            status_parts.append(f"Detected {num_waste_items} waste item(s): {', '.join(waste_summary_parts)}. ({object_list_str})")

        if num_humans > 0:
            plural = "s" if num_humans > 1 else ""
            status_parts.append(f"Detected {num_humans} person{plural}")

        if not status_parts:
            status = "No classifiable objects or people detected."
        else:
            status = ". ".join(status_parts) + "."


        print(f"Processed response. Status: {status}")
        return validated_boxes, status # Return all validated boxes (incl. humans)

    except Exception as e:
        print(f"Critical Error during Gemini API call or processing: {e}")
        traceback.print_exc()
        error_details = f"{type(e).__name__}"
        # Check for API feedback if possible
        # if hasattr(e, 'response') and hasattr(e.response, 'prompt_feedback'):
        #     error_details += f" - Feedback: {e.response.prompt_feedback}"
        return None, f"Error: AI communication failed. ({error_details})"


# --- Flask Routes ---
@app.route('/')
def index():
    """Serves the main landing page."""
    return render_template('index.html')

@app.route('/camera')
def camera_page():
    """Serves the camera detection interface page."""
    return render_template('camera.html')

@app.route('/thank_you')
def thank_you_page():
    """Serves the page shown after the camera session times out."""
    return render_template('thank_you.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    """Receives image frame data, classifies waste/people, returns results."""
    try:
        start_route_time = time.time()
        data = request.get_json()
        if not data or 'image_data' not in data:
            print("Error: Request received without image_data.")
            return jsonify({"error": "Missing image_data", "status": "Client error: No image data received."}), 400

        image_data_url = data['image_data']

        # Decode Base64 Image Data
        try:
            header, encoded = image_data_url.split(",", 1)
            image_bytes = base64.b64decode(encoded)
            mime_type = 'image/jpeg' # Default
            if header.startswith('data:image/'):
                potential_mime = header.split(';')[0].split(':')[1]
                if potential_mime in ['jpeg', 'png', 'webp']:
                    mime_type = f'image/{potential_mime}'
        except (ValueError, base64.binascii.Error, IndexError) as e:
             print(f"Error decoding base64 image data: {e}")
             return jsonify({"error": "Invalid image data format", "status": "Client error: Bad image format."}), 400

        # --- Image Classification and Processing ---
        print("Sending frame to classification...")
        ai_start_time = time.time()
        # classify_waste now returns ALL detected boxes (waste + human)
        all_detected_boxes, status_message = classify_waste(image_bytes)
        ai_time = time.time() - ai_start_time
        print(f"Gemini classification call took {ai_time:.2f} seconds.")

        processed_image_bytes = image_bytes
        # `boxes_found` now means "waste boxes found" for UI clarity
        waste_boxes_found = False
        # `object_details_list` should ONLY contain waste items for the sidebar
        waste_object_details_list = []

        if all_detected_boxes is None:
            # Error during classification
            print(f"Classification failed. Status: {status_message}")
        elif not all_detected_boxes:
            # Nothing found (no waste, no humans)
            print("Classification successful, nothing detected.")
        else:
            # Separate waste from humans
            waste_items = [box for box in all_detected_boxes if box.label != 'Human']
            human_items = [box for box in all_detected_boxes if box.label == 'Human']

            if waste_items:
                waste_boxes_found = True
                waste_object_details_list = [
                    {"name": getattr(box, 'object_name', 'Unknown').title(), "classification": box.label}
                    for box in waste_items
                ]

            print(f"Found {len(waste_items)} waste items and {len(human_items)} humans.")
            print("Drawing boxes for all detected items...")
            draw_start_time = time.time()
            # Draw boxes for ALL items (waste + human) on the image
            drawn_bytes = draw_bounding_boxes(image_bytes, all_detected_boxes)
            draw_time = time.time() - draw_start_time
            print(f"Drawing boxes took {draw_time:.2f} seconds.")

            if drawn_bytes != image_bytes:
                 processed_image_bytes = drawn_bytes
                 print("Successfully drew bounding boxes.")
            else:
                 print("Drawing boxes returned original image (error or no change).")

        # --- Prepare Response ---
        processed_image_base64 = base64.b64encode(processed_image_bytes).decode('utf-8')
        result_image_data_url = f"data:{mime_type};base64,{processed_image_base64}"

        total_route_time = time.time() - start_route_time
        print(f"/process_frame route execution time: {total_route_time:.2f} seconds.")
        print(f"Sending response. Status: '{status_message}', Waste Boxes Found: {waste_boxes_found}")

        # Return status message (covers waste & humans), image with all boxes,
        # waste_boxes_found flag, and ONLY waste details for the sidebar.
        return jsonify({
            "status": status_message,
            "processed_image_data": result_image_data_url,
            "boxes_found": waste_boxes_found, # This flag now specifically means *waste* boxes
            "object_details": waste_object_details_list # Only waste items for the list
        })

    except Exception as e:
        print(f"Critical Error in /process_frame endpoint: {e}")
        traceback.print_exc()
        return jsonify({"error": "An internal server error occurred processing the frame.", "status": "Server error: Processing failed unexpectedly."}), 500


# --- Run Application ---
if __name__ == '__main__':
    print("Starting Veridia Vision Flask development server...")
    # Ensure host='0.0.0.0' to accept connections from other devices on the network
    # Port 5000 is default, change if needed
    # Set debug=False for production and use a proper WSGI server like Gunicorn/Waitress
    app.run(debug=True, host='0.0.0.0', port=5000)
