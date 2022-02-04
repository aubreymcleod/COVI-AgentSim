from InterSim import app, linking
import InterSim.api.library.global_vars as gv

import hydra
from omegaconf import DictConfig, OmegaConf

"""
This file runs the server at a given port.
"""

FLASK_PORT = 8081
DEBUG = True


@hydra.main(config_path="../covid19sim/configs/simulation/config.yaml")
def __main__(conf: DictConfig):
    gv.BaseConfig = conf
    app.run(debug=DEBUG, port=FLASK_PORT, host='0.0.0.0')

if __name__ == "__main__":
    __main__()