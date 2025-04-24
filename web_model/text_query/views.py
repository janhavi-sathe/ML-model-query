from flask import render_template, request, jsonify
from . import text_query_bp
import pandas as pd
import os

@text_query_bp.route('/check_status', methods=['GET'])
def check_status():
    # Returns whether the table data has been loaded.
    print(f"return prediction_done: True")
    return jsonify({"prediction_done": True})

@text_query_bp.route('/text_data', methods=['GET'])
def index():
    return render_template('text_query_form.html')

# Query API (supports GET & POST)
@text_query_bp.route('/text_data/query', methods=['GET', 'POST'])
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
        text_data_path = os.path.join(base_path, "..", "text_data.csv")

        # Load text_data.csv
        text_data_df = pd.read_csv(text_data_path, index_col=0)

        # Get the index that meets the conditions
        matched_indices = text_data_df.loc[text_data_df['pred_labels'] == category, ['text', 'main_words', 'pred_labels', 'prediction']]
        
        # Query Filter
        total_results = len(matched_indices)

        if total_results == 0:
            return jsonify({"error": f"No data found for class {category}."}), 404
        
        # Calculate the paging range
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        paginated_results = matched_indices[start_idx:end_idx]  # Get the corresponding data

        # Formatting data (text index)
        from collections import OrderedDict
        formatted_results = [
            OrderedDict(
                {"Index": int(idx), "Text": row.iloc[0], "Main Words": row.iloc[1], "Class": row.iloc[2], "Prediction": row.iloc[3]}
            )
            for idx, row in paginated_results.iterrows()
        ]
        print("formatted_results: ", formatted_results)

        return jsonify({
            "category": category,
            "total_results": total_results,
            "total_pages": (total_results // per_page) + (1 if total_results % per_page > 0 else 0),
            "current_page": page,
            "per_page": per_page,
            "results": formatted_results
        })

    except FileNotFoundError as e:
        return jsonify({"error": f"File not found: {str(e)}. Please confirm that the file exists at {text_data_path}."}), 500
