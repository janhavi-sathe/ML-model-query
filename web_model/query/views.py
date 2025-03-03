from typing import OrderedDict
from flask import render_template, request, jsonify
import numpy as np
import os
from . import query_bp

# 查詢畫面（首頁）
@query_bp.route('/', methods=['GET'])
def index():
    return render_template('query_form.html')

# 查詢 API（支援 GET & POST）
@query_bp.route('/query', methods=['GET', 'POST'])
def query():
    # 取得類別值
    if request.method == "POST":
        category = request.json.get("class_value")
        page = request.json.get("page", 1)  # 預設為第 1 頁
        per_page = request.json.get("per_page", 10)  # 預設每頁顯示 10 筆
    else:
        category = request.args.get("class_value")
        page = int(request.args.get("page", 1))  # 預設為第 1 頁
        per_page = int(request.args.get("per_page", 10))  # 預設每頁 10 筆


    try:
        category = int(category)
        if category not in range(0, 5):  # ✅ 限制 0~4
            return jsonify({"error": "Please enter a valid class value (0 to 4)."}), 400
    except (TypeError, ValueError):
        return jsonify({"error": "The class value must be a number."}), 400

    print(f"📥 查詢類別: {category}, 頁數: {page}, 每頁顯示: {per_page}")

    try:
        # 取得檔案路徑
        base_path = os.path.dirname(os.path.abspath(__file__))
        X_test_path = os.path.join(base_path, "..", "X_test.npy")
        y_pred_path = os.path.join(base_path, "..", "y_pred.npy")

        # 載入檔案
        X_test_loaded = np.load(X_test_path)
        y_pred_loaded = np.load(y_pred_path)

        print(f"✅ 成功載入 X_test.npy 和 y_pred.npy")
        print(f"📊 X_test 大小: {X_test_loaded.shape}")
        print(f"📊 y_pred 大小: {y_pred_loaded.shape}")

        # 取得符合條件的索引
        matched_indices = np.where(y_pred_loaded == category)[0]

        # 查詢篩選
        filtered_results = X_test_loaded[y_pred_loaded == category]
        total_results = len(filtered_results)  # 全部資料數量

        if total_results == 0:
            return jsonify({"error": f"沒有找到類別 {category} 的資料"}), 404
        
        # ✅ 計算分頁範圍
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        paginated_results = filtered_results[start_idx:end_idx]
        paginated_indices = matched_indices[start_idx:end_idx]  # 取得對應的 index

        # ✅ 加入 index 並使用 OrderedDict 確保 Feature 1 ~ Feature N 順序正確
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
