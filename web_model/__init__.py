from flask import Flask
from web_model.query import query_bp
from web_model.image_query import image_query_bp  # ✅ 加入圖像分類查詢
from web_model.index import index_bp 
from .trainingRL import train_model  # ✅ 改為載入模型
from .imageRL import predict_and_save  # ✅ 改為載入圖像模型
import threading

def create_app(debug=False):

  app = Flask(__name__)
  app.debug = debug

  def load_tabular_data():
    """模擬載入表格數據"""
    train_model()        # 載入數據分類模型

  def load_image_data():
    """模擬載入影像數據"""
    predict_and_save()  # 載入圖像分類模型

  # 伺服器啟動時自動載入數據
  threading.Thread(target=load_tabular_data, daemon=True).start()
  threading.Thread(target=load_image_data, daemon=True).start()
  

  '''# ✅ 只載入模型，不重新訓練
  train_model()        # 載入數據分類模型
  predict_and_save()  # 載入圖像分類模型
  print("123")'''

  # 註冊 Blueprint
  app.register_blueprint(index_bp, url_prefix='/')
  app.register_blueprint(query_bp, url_prefix='/query')
  app.register_blueprint(image_query_bp, url_prefix='/image_query')

  return app