from flask import Flask
from web_model.query import query_bp
from web_model.image_query import image_query_bp  # ✅ 加入圖像分類查詢
from .trainingRL import train_model

def create_app(debug=False):

  app = Flask(__name__)
  app.debug = debug

  app.register_blueprint(query_bp)

  # 執行訓練函數
  train_model()
  
  return app