# -*- coding: utf-8 -*-

"""A WSGI-compliant python module for running the web application for :mod:`seffnet`."""

from seffnet.web import api, create_app

app = create_app()
app.register_blueprint(api)

if __name__ == '__main__':
    app.run()
