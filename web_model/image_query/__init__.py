from flask import Blueprint

image_query_bp = Blueprint("image_query_bp",
                    __name__,
                    template_folder='templates')

from . import views