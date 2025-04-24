from flask import Blueprint

text_query_bp = Blueprint("text_query_bp",
                    __name__,
                    template_folder='templates')

from . import views