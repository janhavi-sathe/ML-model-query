from flask import Blueprint

query_bp = Blueprint("query_bp",
                    __name__,
                    template_folder='templates',
                    static_folder='static',
                    static_url_path='/query/static')

from . import views