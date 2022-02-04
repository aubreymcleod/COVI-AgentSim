from flask import Blueprint, session, render_template, redirect, request, make_response

from InterSim.api.config import config
from InterSim.api.sim import sim

api = Blueprint('api', __name__)

api.register_blueprint(config, url_prefix='/config')
api.register_blueprint(sim, url_prefix='/sim')