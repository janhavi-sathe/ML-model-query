from flask import Blueprint, render_template

index_bp = Blueprint("index_bp", __name__, template_folder="templates")

@index_bp.route('/')
def index():
    return render_template('index.html')  # 確保 `index.html` 存在
