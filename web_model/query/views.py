from flask import render_template, request, jsonify
import numpy as np
import os
from . import query_bp
from .. import trainingRL

@query_bp.route('/check_status', methods=['GET'])
def check_status():
    # Returns whether the table data has been loaded.
    print(f"return prediction_done: {trainingRL.prediction_done}")
    return jsonify({"prediction_done": trainingRL.prediction_done})

# Search screen (Home page)
@query_bp.route('/tabular_data', methods=['GET'])
def index():
    return render_template('query_form.html')

# Query API (supports GET & POST)
@query_bp.route('/tabular_data/query', methods=['GET', 'POST'])
def query():
    # Get category value
    if request.method == "POST":
        category = request.json.get("class_value")
        page = request.json.get("page", 1)  # Default is Page 1
        per_page = request.json.get("per_page", 10)  # Default display is 10 records per page
    else:
        category = request.args.get("class_value")
        page = int(request.args.get("page", 1))  # Default is Page 1
        per_page = int(request.args.get("per_page", 10))  # Default display is 10 records per page


    try:
        category = int(category)
        if category not in range(0, 5):  # Limit 0~4
            return jsonify({"error": "Please enter a valid class value (0 to 4)."}), 400
    except (TypeError, ValueError):
        return jsonify({"error": "The class value must be a number."}), 400

    try:
        # Get the file path
        base_path = os.path.dirname(os.path.abspath(__file__))
        X_test_path = os.path.join(base_path, "..", "X_test.npy")
        y_pred_path = os.path.join(base_path, "..", "y_pred.npy")

        # Loading File
        X_test_loaded = np.load(X_test_path)
        y_pred_loaded = np.load(y_pred_path)

        # Get the index that meets the conditions
        matched_indices = np.where(y_pred_loaded == category)[0]

        # Query Filter
        filtered_results = X_test_loaded[y_pred_loaded == category]
        total_results = len(filtered_results)  # Total data quantity

        if total_results == 0:
            return jsonify({"error": f"No data found for class {category}."}), 404
        
        # Calculate the paging range
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        paginated_results = filtered_results[start_idx:end_idx]
        paginated_indices = matched_indices[start_idx:end_idx]  # Get the corresponding index

        # Add index and use OrderedDict to ensure that Feature 1 ~ Feature N are in the correct order
        from collections import OrderedDict
        formatted_results = [
            OrderedDict(
                {"Index": int(idx), **{f"Feature {i+1}": value for i, value in enumerate(row)}}
            )
            for idx, row in zip(paginated_indices, paginated_results)
        ]

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
