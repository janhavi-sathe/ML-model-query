from flask import Blueprint

image_query_bp = Blueprint("image_query_bp",
                    __name__,
                    template_folder='templates',
                    static_folder='static',
                    static_url_path='/image_query/static')

from . import views