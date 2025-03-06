'''from flask import Blueprint

index_bp = Blueprint("index_bp",
                    __name__,
                    template_folder='templates',
                    static_folder='static',
                    static_url_path='/index/static')

from . import views'''

from flask import Blueprint, render_template

index_bp = Blueprint("index_bp", __name__, template_folder="templates")

@index_bp.route('/')
def index():
    print("已印出首頁！")
    return render_template('index.html')  # ✅ 確保 `index.html` 存在
