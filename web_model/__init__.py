from flask import Flask
from web_model.query import query_bp
from web_model.image_query import image_query_bp  # 加入圖像分類查詢
from web_model.text_query import text_query_bp
from web_model.index import index_bp 
from .trainingRL import train_model  # 改為載入模型
from .imageRL import predict_and_save  # 改為載入圖像模型
import threading
import time
from config import Config

def create_app(config_class=Config,debug=False):

  app = Flask(__name__)
  app.config.from_prefixed_env("FLASK_")
  app.config.from_object(config_class)
  app.debug = debug

  def load_tabular_data():
    # 模擬載入表格數據
    train_model()        # 載入數據分類模型

  def load_image_data():
    # 模擬載入影像數據
    predict_and_save()  # 載入圖像分類模型

  def load_text_data():
    # 模擬載入文本數據
    time.delay(5)

  # 伺服器啟動時自動載入數據
  threading.Thread(target=load_tabular_data, daemon=True).start()
  threading.Thread(target=load_image_data, daemon=True).start()
  #threading.Thread(target=load_text_data, daemon=True).start()

  # 註冊 Blueprint
  app.register_blueprint(index_bp, url_prefix='/')
  app.register_blueprint(query_bp, url_prefix='/query')
  app.register_blueprint(image_query_bp, url_prefix='/image_query')
  app.register_blueprint(text_query_bp, url_prefix='/text_query')

  return app