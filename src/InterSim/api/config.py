from flask import Blueprint, request, jsonify, Response
import InterSim.api.library.global_vars as gv

from omegaconf import OmegaConf
import logging

"""
This file defines the launchpage through Blueprints, and implements any needed background logic.
"""
config = Blueprint('config', __name__)

# routing
@config.route('/', methods=['GET'])
def get():
    """
    get the CurrentConfig
    """
    if gv.CurrentConfig is None:
        logging.warning("Current Config not set, setting from base...")
        gv.CurrentConfig = gv.BaseConfig
    message = OmegaConf.to_container(gv.CurrentConfig, resolve=True)
    resp = jsonify({
        'response': message,
        'status': 200,
        'mimetype': 'application/json'})
    return resp

@config.route('/reset', methods=['GET'])
def reset():
    """
    reset the CurrentConfig to the default value.
    """
    logging.info("Reset Current Config")
    gv.CurrentConfig = gv.BaseConfig
    return get()

@config.route('/', methods=['PUT'])
def put():
    """
    Take Json requests to edit config.
    """
    try:
        content = request.get_json()
        gv.CurrentConfig = OmegaConf.create({**gv.CurrentConfig, **content})
        logging.info("Applying update to current config")
        response = Response('OK', status=200)
        return response
    except Exception:
        return Response('Failure', status=500)