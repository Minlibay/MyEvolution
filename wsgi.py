from a2wsgi import ASGIMiddleware

from src.web.app import app

application = ASGIMiddleware(app)
