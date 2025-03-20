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
    """回傳影像數據是否載入完成"""
    print(f"return image_prediction_done: {imageRL.image_prediction_done}")
    return jsonify({"prediction_done": imageRL.image_prediction_done})

@image_query_bp.route('/image_data', methods=['GET'])
def index():
    return render_template('image_query_form.html')

# 查詢 API（支援 GET & POST）
@image_query_bp.route('/image_data/query', methods=['GET', 'POST'])
def query():
    # 取得類別值
    if request.method == "POST":
        category = request.json.get("class_value")
        page = request.json.get("page", 1)  # 預設為第 1 頁
        per_page = 8  # 修改為每頁顯示 8 張圖
    else:
        category = request.args.get("class_value")
        page = int(request.args.get("page", 1))  # 預設為第 1 頁
        per_page = 8  # 修改為每頁顯示 8 張圖

    try:
        category = int(category)
        if category not in range(0, 5):  # 限制 0~4
            return jsonify({"error": "Please enter a valid class value (0 to 4)."}), 400
    except (TypeError, ValueError):
        return jsonify({"error": "The class value must be a number."}), 400

    try:
        # 取得檔案路徑
        base_path = os.path.dirname(os.path.abspath(__file__))
        X_test_path = os.path.join(base_path, "..", "X_test_image.npz")
        y_pred_path = os.path.join(base_path, "..", "y_pred_image.npy")

        # 載入 X_test_image.npz
        with np.load(X_test_path) as data:
            print("X_test_image.npz keys:", list(data.keys()))  # 確保包含 'x'
            X_test_loaded = data["x"]  # 讀取圖片數據

        # 載入檔案
        y_pred_loaded = np.load(y_pred_path)

        y_pred_labels = y_pred_loaded.argmax(axis=1)

        # 取得符合條件的索引
        matched_indices = np.where(y_pred_labels == category)[0]

        # 查詢篩選
        total_results = len(matched_indices)

        if total_results == 0:
            return jsonify({"error": f"No data found for class {category}."}), 404
        
        # 計算分頁範圍
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        #paginated_results = filtered_results[start_idx:end_idx]
        paginated_indices = matched_indices[start_idx:end_idx]  # 取得對應的 index

        # 格式化數據（圖片索引）
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
        # 獲取當前檔案的基礎路徑
        base_path = os.path.dirname(os.path.abspath(__file__))
        X_test_image_path = os.path.join(base_path, "..", "X_test_image.npz")

        # 加載 .npz 文件
        with np.load(X_test_image_path) as data:
            print("X_test_image_path exists, keys:", list(data.keys()))
            X_test_loaded = data["x"]

        # 確保索引有效
        if index < 0 or index >= len(X_test_loaded):
            return jsonify({"error": "Image index out of range."}), 404

        # 讀取圖片數據
        image_array = X_test_loaded[index]
        print(f"image_array shape: {image_array.shape}, dtype: {image_array.dtype}")

        # 確保數據範圍是 0-255 並且是 uint8
        if image_array.dtype != np.uint8:
            image_array = (image_array * 255).astype(np.uint8)

        # 確保維度是 (H, W, 3)
        if len(image_array.shape) == 2:  # 如果是灰階，轉 RGB
            image_array = np.stack([image_array] * 3, axis=-1)

        # 嘗試創建 PIL 圖片
        try:
            print(f"Creating image from array with shape: {image_array.shape}, dtype: {image_array.dtype}")
            image = Image.fromarray(image_array)
        except Exception as e:
            print(f"Error creating image: {e}")
            return jsonify({"error": f"Failed to create image: {str(e)}"}), 500

        # 獲取 views.py 的當前目錄
        base_dir = os.path.dirname(os.path.abspath(__file__))
        # 存到 Flask static/images 目錄
        img_dir = os.path.join(base_dir, "static", "images")
        os.makedirs(img_dir, exist_ok=True)  # 確保目錄存在

        #img_path = os.path.join(img_dir, f"temp_{index}.png")
        #image.save(img_path)
        img_path = os.path.join(img_dir, f"temp_{index}.jpg")
        image = image.convert("RGB")  # 確保不是 PNG（透明背景）
        image.save(img_path, "JPEG", quality=85)  # 壓縮圖片

        print(f"Image saved at: {img_path}")

        # 確保圖片已經儲存
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file {img_path} not found after saving.")
        
        # 先傳送圖片，5 秒後刪除
        delayed_remove(img_path, delay=5)

        # 返回圖片文件
        return send_file(img_path, mimetype='image/jpeg')

    except FileNotFoundError:
        return jsonify({"error": "File X_test_image.npz not found."}), 500
    except KeyError:
        return jsonify({"error": "Invalid .npz file structure. Expected 'x' as key."}), 500
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500
    
def delayed_remove(file_path, delay=5):
    # 延遲刪除檔案，確保 Flask 已經成功傳送圖片
    def remove():
        time.sleep(delay)  # 延遲刪除，確保傳輸完成
        try:
            os.remove(file_path)
            print(f"Deleted temp file: {file_path}")
        except Exception as e:
            print(f"Failed to delete {file_path}: {e}")

    threading.Thread(target=remove, daemon=True).start()  # 背景執行刪除

@image_query_bp.route('/find_similar', methods=['GET'])
def find_similar_images():
    try:
        image_index = request.args.get("index")
        if image_index is None:
            return jsonify({"error": "Missing image index parameter"}), 400

        image_index = int(image_index)

        # 讀取 X_test.npy（圖片特徵）
        base_path = os.path.dirname(os.path.abspath(__file__))
        X_test_path = os.path.join(base_path, "..", "X_test.npy")

        X_test_loaded = np.load(X_test_path)  # shape: (num_samples, feature_dim)

        if image_index < 0 or image_index >= len(X_test_loaded):
            return jsonify({"error": "Image index out of range"}), 404

        # 取得指定圖片的特徵向量
        query_image_feature = X_test_loaded[image_index]

        # 計算歐幾里得距離
        distances = np.linalg.norm(X_test_loaded - query_image_feature, axis=1)

        # 取得最相似的 8 張圖片（排除自己）
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
