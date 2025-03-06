from flask import Blueprint

query_bp = Blueprint("query_bp",
                    __name__,
                    template_folder='templates')

from . import views