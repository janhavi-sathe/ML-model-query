from flask import render_template, request, jsonify
from . import text_query_bp
import pandas as pd
import numpy as np
import os
import json, csv

snips_id2label = {0: 'AddToPlaylist', 1: 'BookRestaurant', 2: 'GetWeather', 3: 'PlayMusic', 4: 'RateBook', 5: 'SearchCreativeWork', 6: 'SearchScreeningEvent'}

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
        per_page = int(request.args.get("per_page", 8))   # Default display 8 texts per page
    else:
        category = request.args.get("class_value")
        page = int(request.args.get("page", 1))  # Default is Page 1
        per_page = int(request.args.get("per_page", 8))  # Default display 8 texts per page

    try:
        category = int(category)
        
        if category not in range(0, 7):  # Limit 0~7
            return jsonify({"error": "Please enter a valid class value (0 to 6)."}), 400
    except (TypeError, ValueError):
        return jsonify({"error": "The class value must be a number."}), 400

    try:
        # Get the file path
        base_path = os.path.dirname(os.path.abspath(__file__))
        text_data_path = os.path.join(base_path, "..", "snips_text_data.csv")

        # Load snips_text_data.csv
        text_data_df = pd.read_csv(text_data_path)
        # Get the data that meets the conditions
        matched_indices = text_data_df.loc[text_data_df['model-assigned label'] == snips_id2label[category], :] # column ['utterance', 'explanation', 'model-assigned label', 'human-assigned label']
        
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
                {"Index": int(idx), "Text": row.iloc[0], "Human-assigned Label": row.iloc[1] , "Model-assigned Label": row.iloc[2], "Explanation": row.iloc[3]}
            )
            for idx, row in paginated_results.iterrows()
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
        return jsonify({"error": f"File not found: {str(e)}. Please confirm that the file exists at {text_data_path}."}), 500


@text_query_bp.route('text_data/find_similar', methods=['GET'])
def find_similar_images():
    try:
        text_index = request.args.get("index")
        page = int(request.args.get("page", 1))  # Default is Page 1
        per_page = int(request.args.get("per_page", 8))  # Default display 8 texts per page

        if text_index is None:
            return jsonify({"error": "Missing text index parameter"}), 400

        text_index = int(text_index)

        # Read X_test.npy (text embeddings)
        base_path = os.path.dirname(os.path.abspath(__file__))
        sim_mat_path = os.path.join(base_path, "..", "snips_sim_matrix.npy")

        sim_mat_loaded = np.load(sim_mat_path)  

        # Get the file path
        base_path = os.path.dirname(os.path.abspath(__file__))
        text_data_path = os.path.join(base_path, "..", "snips_text_data.csv")

        # Load text_data.csv
        text_data_df = pd.read_csv(text_data_path)

        if text_index < 0 or text_index >= len(text_data_df):
            return jsonify({"error": "Text index out of range"}), 404

        # Get similarities
        similarities = sim_mat_loaded[text_index]

        # Get the TOP_N most similar texts (excluding yourself)
        # TOP_N = 10
        SIMILARITY_THRESHOLD = 0.9
        similar_indices = np.extract(similarities > SIMILARITY_THRESHOLD, similarities)
        
        sorted_indices = [int(idx) for idx in np.argsort(similar_indices)][-1::-1]
        #similar_indices = [int(idx) for idx in sorted_indices][-TOP_N:]
        #print(sorted_indices)

        # Filter data by similar indices
        similar_texts = text_data_df.loc[sorted_indices, :] # columns=['utterance', 'human-assigned label', 'model-assigned label', 'explanation']

        # Query Filter
        total_results = len(similar_texts)

        if total_results == 0:
            return jsonify({"error": f"No similar text found for index {text_index}."}), 404
        
        # Calculate the paging range
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        paginated_results = similar_texts[start_idx:end_idx]  # Get the corresponding data

        # Formatting data (text index)
        from collections import OrderedDict
        formatted_results = [
            OrderedDict(
                {"Index": int(idx), "Text": row.iloc[0], "Human-assigned Label": row.iloc[1] , "Model-assigned Label": row.iloc[2], "Explanation": row.iloc[3]}
            )
            for idx, row in paginated_results.iterrows()
        ]

        return jsonify({
            "query_index": text_index, 
            "total_results": total_results,
            "total_pages": (total_results // per_page) + (1 if total_results % per_page > 0 else 0),
            "current_page": page,
            "per_page": per_page,
            "similar_results": formatted_results
        })

    except ValueError:
        return jsonify({"error": "Invalid text index"}), 400
    except FileNotFoundError:
        return jsonify({"error": "snips_sim_matrix.npy not found"}), 500
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

@text_query_bp.route('text_data/find_keyword', methods=['GET'])
def find_keyword():
    try:
        keyword = request.args.get("keyword").lower()
        if keyword is None:
            return jsonify({"error": "Missing keyword parameter"}), 400

        keyword = keyword.strip()

        page = int(request.args.get("page", 1))  # Default is Page 1
        per_page = int(request.args.get("per_page", 8))  # Deafult display 8 texts per page

        # Get the file path
        base_path = os.path.dirname(os.path.abspath(__file__))
        text_data_path = os.path.join(base_path, "..", "snips_text_data.csv")

        # Load text_data.csv
        text_data_df = pd.read_csv(text_data_path)

        # Get texts that contain the keyword
        keyword_data = text_data_df.loc[text_data_df['utterance'].astype(str).str.contains(keyword, na=False), :] #columns=['utterance', 'human-assigned label', 'model-assigned label', 'explanation']

        # Query Filter
        total_results = len(keyword_data)
        print(total_results)

        if total_results == 0:
            return jsonify({"error": f"No data found containing keyword {keyword}."}), 204
        
        # Calculate the paging range
        start_idx = (page - 1) * per_page
        print(start_idx)
        end_idx = start_idx + per_page
        print(end_idx)
        paginated_results = keyword_data[start_idx:end_idx]  # Get the corresponding data

        # Formatting data (text index)
        from collections import OrderedDict
        formatted_results = [
            OrderedDict(
                {"Index": int(idx), "Text": row.iloc[0], "Human-assigned Label": row.iloc[1] , "Model-assigned Label": row.iloc[2], "Explanation": row.iloc[3]}
            )
            for idx, row in paginated_results.iterrows()
        ]

        return jsonify({
            "query_word": keyword, 
            "total_results": total_results,
            "total_pages": (total_results // per_page) + (1 if total_results % per_page > 0 else 0),
            "current_page": page,
            "per_page": per_page,
            "results": formatted_results
        })

    except ValueError:
        return jsonify({"error": "Invalid text index"}), 400
    except FileNotFoundError:
        return jsonify({"error": "snips_text_data.csv not found"}), 500
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500
    
@text_query_bp.route('/feedback', methods=['POST'])
def feedback():
    try:
        feedback_data = request.json.get("feedback")
        timestamp = request.json.get("timestamp")

        if feedback_data is None:
            return jsonify({"error": "Missing feedback parameter"}), 400
        
        base_path = os.path.dirname(os.path.abspath(__file__))
        feedback_path = os.path.join(base_path, "..", "feedback.csv")

        # clean feedback
        feedback_data = feedback_data.strip()
        feedback_data = feedback_data.replace("\n", " ")

        with open(feedback_path, 'a') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow([timestamp, feedback_data])
        
        return jsonify({"success": "Feedback submitted successfully."}), 200
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500