# -*- coding: utf-8 -*-

"""A WSGI-compliant python module for running the web application for ``se_kge``."""

from se_kge.web import api, create_app

app = create_app()
app.register_blueprint(api)
app.run()
