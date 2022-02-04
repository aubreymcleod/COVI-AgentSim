from flask import render_template
from InterSim import app


# import api routes
from InterSim.api.api import api

# import view routes
from InterSim.views.home import home


# link api routes
app.register_blueprint(api, url_prefix='/api')

# link view routes
app.register_blueprint(home)


"""
display a catch-all 404 error page whenever our app hits an unknown route.
"""
@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html', title='404'), 404
