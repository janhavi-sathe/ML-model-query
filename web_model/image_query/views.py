from flask import render_template, request, jsonify, send_file
import numpy as np
import os
import time
import threading
from PIL import Image
from . import image_query_bp
from .. import imageRL

@image_query_bp.route('/check_status', methods=['GET'])
def check_image_status():
    """Returns whether the image data has been loaded."""
    print(f"return image_prediction_done: {imageRL.image_prediction_done}")
    return jsonify({"prediction_done": imageRL.image_prediction_done})

@image_query_bp.route('/image_data', methods=['GET'])
def index():
    return render_template('image_query_form.html')

# Query API (supports GET & POST)
@image_query_bp.route('/image_data/query', methods=['GET', 'POST'])
def query():
    # Get category value
    if request.method == "POST":
        category = request.json.get("class_value")
        page = request.json.get("page", 1)  # Default is Page 1
        per_page = 8  # Modified to display 8 pictures per page
    else:
        category = request.args.get("class_value")
        page = int(request.args.get("page", 1))  # Default is Page 1
        per_page = 8  # Modified to display 8 pictures per page

    try:
        category = int(category)
        if category not in range(0, 5):  # Limit 0~4
            return jsonify({"error": "Please enter a valid class value (0 to 4)."}), 400
    except (TypeError, ValueError):
        return jsonify({"error": "The class value must be a number."}), 400

    try:
        # Get the file path
        base_path = os.path.dirname(os.path.abspath(__file__))
        X_test_path = os.path.join(base_path, "..", "X_test_image.npz")
        y_pred_path = os.path.join(base_path, "..", "y_pred_image.npy")

        # Load X_test_image.npz
        with np.load(X_test_path) as data:
            print("X_test_image.npz keys:", list(data.keys()))  # Make sure to include 'x'
            X_test_loaded = data["x"]  # Reading image data

        # Loading File
        y_pred_loaded = np.load(y_pred_path)

        y_pred_labels = y_pred_loaded.argmax(axis=1)

        # Get the index that meets the conditions
        matched_indices = np.where(y_pred_labels == category)[0]

        # Query Filter
        total_results = len(matched_indices)

        if total_results == 0:
            return jsonify({"error": f"No data found for class {category}."}), 404
        
        # Calculate the paging range
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        #paginated_results = filtered_results[start_idx:end_idx]
        paginated_indices = matched_indices[start_idx:end_idx]  # Get the corresponding index

        # Formatting data (image index)
        formatted_results = [{"Index": int(idx)} for idx in paginated_indices]

        return jsonify({
            "category": category,
            "total_results": total_results,
            "total_pages": (total_results // per_page) + (1 if total_results % per_page > 0 else 0),
            "current_page": page,
            "per_page": per_page,
            "results": formatted_results
        })

    except FileNotFoundError as e:
        return jsonify({"error": f"File not found: {str(e)}. Please confirm that the file exists at {X_test_path}."}), 500

@image_query_bp.route('/image/<int:index>', methods=['GET'])
def get_image(index):
    try:
        # Get the base path of the current archive
        base_path = os.path.dirname(os.path.abspath(__file__))
        X_test_image_path = os.path.join(base_path, "..", "X_test_image.npz")

        # Loading .npz files
        with np.load(X_test_image_path) as data:
            print("X_test_image_path exists, keys:", list(data.keys()))
            X_test_loaded = data["x"]

        # Make sure the index is valid
        if index < 0 or index >= len(X_test_loaded):
            return jsonify({"error": "Image index out of range."}), 404

        # Reading image data
        image_array = X_test_loaded[index]
        print(f"image_array shape: {image_array.shape}, dtype: {image_array.dtype}")

        # Make sure the data range is 0-255 and is uint8
        if image_array.dtype != np.uint8:
            image_array = (image_array * 255).astype(np.uint8)

        # Make sure the dimensions are (H, W, 3)
        if len(image_array.shape) == 2:  # If grayscale, convert to RGB
            image_array = np.stack([image_array] * 3, axis=-1)

        # Try creating a PIL image
        try:
            print(f"Creating image from array with shape: {image_array.shape}, dtype: {image_array.dtype}")
            image = Image.fromarray(image_array)
        except Exception as e:
            print(f"Error creating image: {e}")
            return jsonify({"error": f"Failed to create image: {str(e)}"}), 500

        # Get the current directory of views.py
        base_dir = os.path.dirname(os.path.abspath(__file__))
        # Save to the Flask static/images directory
        img_dir = os.path.join(base_dir, "static", "images")
        os.makedirs(img_dir, exist_ok=True)  # Make sure the directory exists

        #img_path = os.path.join(img_dir, f"temp_{index}.png")
        #image.save(img_path)
        img_path = os.path.join(img_dir, f"temp_{index}.jpg")
        image = image.convert("RGB")  # Make sure it is not a PNG (transparent background)
        image.save(img_path, "JPEG", quality=85)  # Compress images

        print(f"Image saved at: {img_path}")

        # Make sure the image is saved
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file {img_path} not found after saving.")
        
        # Send the image first, then delete it after 5 seconds
        delayed_remove(img_path, delay=5)

        # Return image file
        return send_file(img_path, mimetype='image/jpeg')

    except FileNotFoundError:
        return jsonify({"error": "File X_test_image.npz not found."}), 500
    except KeyError:
        return jsonify({"error": "Invalid .npz file structure. Expected 'x' as key."}), 500
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500
    
def delayed_remove(file_path, delay=5):
    # Delay the file deletion to ensure that Flask has successfully sent the image
    def remove():
        time.sleep(delay)  # Delay deletion to ensure transfer is complete
        try:
            os.remove(file_path)
            print(f"Deleted temp file: {file_path}")
        except Exception as e:
            print(f"Failed to delete {file_path}: {e}")

    threading.Thread(target=remove, daemon=True).start()  # Background execution removal

@image_query_bp.route('/find_similar', methods=['GET'])
def find_similar_images():
    try:
        image_index = request.args.get("index")
        if image_index is None:
            return jsonify({"error": "Missing image index parameter"}), 400

        image_index = int(image_index)

        # Read X_test.npy (image features)
        base_path = os.path.dirname(os.path.abspath(__file__))
        X_test_path = os.path.join(base_path, "..", "X_test.npy")

        X_test_loaded = np.load(X_test_path)  # shape: (num_samples, feature_dim)

        if image_index < 0 or image_index >= len(X_test_loaded):
            return jsonify({"error": "Image index out of range"}), 404

        # Get the feature vector of the specified image
        query_image_feature = X_test_loaded[image_index]

        # Calculate Euclidean distance
        distances = np.linalg.norm(X_test_loaded - query_image_feature, axis=1)

        # Get the 8 most similar images (excluding yourself)
        sorted_indices = np.argsort(distances)
        similar_indices = [int(idx) for idx in sorted_indices if idx != image_index][:8]

        return jsonify({
            "query_index": image_index,
            "similar_images": [{"Index": idx} for idx in similar_indices]
        })

    except ValueError:
        return jsonify({"error": "Invalid image index"}), 400
    except FileNotFoundError:
        return jsonify({"error": "X_test.npy not found"}), 500
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500
