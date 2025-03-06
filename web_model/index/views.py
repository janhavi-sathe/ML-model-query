from flask import render_template
from . import index_bp

# 查詢畫面（首頁）
@index_bp.route('/', methods=['GET'])
def index():
    return render_template('index.html')