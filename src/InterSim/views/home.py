from flask import Blueprint, session, render_template, redirect, request, make_response

"""
This file defines the launchpage through Blueprints, and implements any needed background logic.
"""

home = Blueprint('home', __name__, template_folder='templates')


@home.route('/', methods=['GET', 'POST'])
def launch():
    resp = make_response(render_template('404.html'))
    return resp
