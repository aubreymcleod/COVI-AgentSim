from flask import Flask
import os

"""
This file defines global variables and config values
"""

package_dir = os.path.dirname(
    os.path.abspath(__file__)
)

templates = os.path.join(
    package_dir, "templates"
)

app = Flask('This is an application for handling Interactive Modeling with CoviSim', template_folder=templates)
