import datetime
from flask import render_template, request, jsonify
from . import text_query_bp
import pandas as pd
import numpy as np
import os
import json, csv
from config import Config, LoggerConfig
import logging 

EXPERIMENT_GROUP = Config.EXPERIMENT_GROUP

DATA_FILE = Config.TEXT_DATA_FILENAME
SIMILARITY_FILE = Config.TEXT_SIMILARITY_FILENAME

snips_id2label = Config.TEXT_ID2LABEL
SELECTED = Config.RLA_SELECTED_TEXTS

TOP_N = Config.TOP_N_SIMILAR_TEXTS
SIMILARITY_THRESHOLD = Config.TEXT_SIMILARITY_THRESHOLD

DEFAULT_PER_PAGE = Config.DEFAULT_PER_PAGE

filename = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".log"
filepath = os.path.join(LoggerConfig.basepath, filename)

logging.basicConfig(filename=filepath,
                    encoding=LoggerConfig.encoding,
                    filemode=LoggerConfig.filemode,
                    format=LoggerConfig.format,
                    style=LoggerConfig.style,
                    datefmt=LoggerConfig.datefmt,
                    level=LoggerConfig.level
                )

@text_query_bp.route('/check_status', methods=['GET'])
def check_status():
    # Returns whether the table data has been loaded.
    print(f"return prediction_done: True")
    return jsonify({"prediction_done": True})

def load_text_data():
    try:
        # Get the file path
        base_path = os.path.dirname(os.path.abspath(__file__))
        text_data_path = os.path.join(base_path, "..", DATA_FILE)

        # Load snips_text_data.csv
        text_data_df = pd.read_csv(text_data_path, index_col=0)

        ### sort according to the index
        # text_data_df.sort_index(inplace=True) #by=text_data_df.columns[0],

        if EXPERIMENT_GROUP:
            # Create a Series mapping index to rank (1-based)
            rank_map = {idx: i+1 for i, idx in enumerate(SELECTED)}

            # Assign priority_rank using the rank_map; others get NaN
            text_data_df['priority_rank'] = text_data_df.index.map(rank_map)

            # Sort by priority_rank: selected rows go to top in correct order
            # NaNs (non-selected) will be placed at the end
            text_data_df = text_data_df.sort_values(by='priority_rank', na_position='last')

            # Set featured flag for RLA selected texts
            text_data_df.loc[SELECTED, "featured"] = True

        return text_data_df
    
    except FileNotFoundError as e:
        return jsonify({"error": f"File not found: {str(e)}. Please confirm that the file exists at {DATA_FILE}."}), 500

@text_query_bp.route('/click_data', methods=['POST'])
def click_data():
    log_file = filepath
    with open(log_file, "a") as f:
        f.write(json.dumps(request.json) + "\n")
    return jsonify({"success": True}), 200

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
        per_page = int(request.args.get("per_page", DEFAULT_PER_PAGE))   
    else:
        category = request.args.get("class_value")
        page = int(request.args.get("page", 1))  # Default is Page 1
        per_page = int(request.args.get("per_page", DEFAULT_PER_PAGE))  

    try:
        category = int(category)
        
        if category not in range(0, 7):  # Limit 0~7
            return jsonify({"error": "Please enter a valid class value (0 to 6)."}), 400
    except (TypeError, ValueError):
        return jsonify({"error": "The class value must be a number."}), 400

    try:
        text_data_df = load_text_data()

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
                {"Index": int(idx), "Text": row.iloc[0], "Human-assigned Label": row.iloc[1] , "Model-assigned Label": row.iloc[2], "Explanation": row.iloc[3], "featured": row.iloc[4]}
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
        return jsonify({"error": f"File not found: {str(e)}. Please confirm that the file exists at {DATA_FILE}."}), 500


@text_query_bp.route('text_data/find_similar', methods=['GET'])
def find_similar_texts():
    try:
        # Parse input parameters
        text_index = request.args.get("index")
        page = int(request.args.get("page", 1))  # Default is Page 1
        per_page = int(request.args.get("per_page", DEFAULT_PER_PAGE))  

        if text_index is None:
            return jsonify({"error": "Missing text index parameter"}), 400

        text_index = int(text_index)

        # Read X_test.npy (text embeddings)
        base_path = os.path.dirname(os.path.abspath(__file__))
        sim_mat_path = os.path.join(base_path, "..", SIMILARITY_FILE)

        sim_mat_loaded = np.load(sim_mat_path)  

        # Load text_data.csv file
        text_data_df = load_text_data()

        if text_index < 0 or text_index >= len(text_data_df):
            return jsonify({"error": "Text index out of range"}), 404

        # Get similarities
        similarities = sim_mat_loaded[text_index]

        # Get all the most similar texts (excluding yourself)
        similar_indices = np.extract(similarities > SIMILARITY_THRESHOLD, similarities)
        
        # Sort in order of high -> low similarity
        sorted_indices = [int(idx) for idx in np.argsort(similar_indices)][-1::-1]

        # Filter data by similar indices
        similar_texts = text_data_df.loc[sorted_indices, :] # columns=['utterance', 'human-assigned label', 'model-assigned label', 'explanation']

        # Query Filter
        total_results = len(similar_texts)

        if total_results == 0:
            return jsonify({"error": f"No similar text found for index {text_index}."}), 404
        
        # Limit the number of results to TOP_N
        if total_results > TOP_N:
            total_results = TOP_N
            similar_texts = similar_texts.head(TOP_N)
        
        if EXPERIMENT_GROUP:
            similar_texts.sort_values(by='featured', ascending=False, inplace=True)
        
        # Calculate the paging range
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        paginated_results = similar_texts[start_idx:end_idx]  # Get the corresponding data

        # Formatting data (text index)
        from collections import OrderedDict
        formatted_results = [
            OrderedDict(
                {"Index": int(idx), "Text": row.iloc[0], "Human-assigned Label": row.iloc[1] , "Model-assigned Label": row.iloc[2], "Explanation": row.iloc[3], "featured": row.iloc[4]}
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
        per_page = int(request.args.get("per_page", DEFAULT_PER_PAGE))  

        text_data_df = load_text_data()

        # Get texts that contain the keyword
        keyword_data = text_data_df.loc[text_data_df['utterance'].astype(str).str.contains(keyword, na=False), :] #columns=['utterance', 'human-assigned label', 'model-assigned label', 'explanation']

        # Query Filter
        total_results = len(keyword_data)
        print(total_results)

        if total_results == 0:
            print("No data found containing keyword.")
            return jsonify({"error": f"No data found containing keyword {keyword}"}), 404
        
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
                {"Index": int(idx), "Text": row.iloc[0], "Human-assigned Label": row.iloc[1] , "Model-assigned Label": row.iloc[2], "Explanation": row.iloc[3], "featured": row.iloc[4]}
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