import hydra
import json
import os
import sys

import streamlit as st

from contextlib import contextmanager
from io import StringIO
from omegaconf import DictConfig, OmegaConf
from streamlit import cli as stcli
from streamlit.script_run_context import SCRIPT_RUN_CONTEXT_ATTR_NAME
from threading import current_thread

from covid19sim.interactivity.controlled_run import ControlledSimulation
from InterSim.components.visualizationEngine import visualizer

timesteps = {"1 second": 1.0,
             "10 seconds": 10.0,
             "1 minute": 60.0,
             "10 minutes": 600.0,
             "1 hour": 3600.0,
             "4 hours": 14400.0,
             "12 hours": 43200.0,
             "24 hours": 86400.0}

plotly_config = {"scrollZoom": True, 'modeBarButtonsToRemove': ['zoom'], 'dragmode': 'pan'}


def get_sim(conf: DictConfig):
    sim = ControlledSimulation(conf)
    return sim


def get_config():
    with open("confcache.json", "r") as infile:
        return OmegaConf.create(json.load(infile))

# =======
# Console
# =======
@contextmanager
def st_redirect(src, dst):
    placeholder = st.empty()
    output_func = getattr(placeholder, dst)

    with StringIO() as buffer:
        old_write = src.write

        def new_write(b):
            if getattr(current_thread(), SCRIPT_RUN_CONTEXT_ATTR_NAME, None):
                st.session_state["log"] += b
                buffer.write(b)
                output_func(buffer.getvalue())
            else:
                old_write(b)

        try:
            src.write = new_write
            yield
        finally:
            src.write = old_write


@contextmanager
def st_stdout(dst):
    with st_redirect(sys.stdout, dst):
        yield

@contextmanager
def st_stderr(dst):
    with st_redirect(sys.stderr, dst):
        yield

# ===========
# UI Elements
# ===========
def draw_sidebar(visualization_area):
    """
    st.sidebar.write(f'Simulation date')
    time = st.sidebar.empty()
    time.info(st.session_state["sim"].env.timestamp)

    st.sidebar.write("[SELECTED AGENT NAME HERE]")
    schedule = st.sidebar.empty()
    schedule.info("[SCHEDULE WIDGET HERE]")

    speed_selection = st.sidebar.select_slider("stepsize", options=timesteps.keys())
    step_btn = st.sidebar.empty()
    if step_btn.button('Step forward', disabled=st.session_state["sim"].env.now == st.session_state["sim"].end_time):
        with st.spinner(text="Please wait..."):
            msg = st.session_state["sim"].auto_step(timesteps[speed_selection])
            #toastr pop the msg
        time.info(st.session_state["sim"].env.timestamp)
        st.session_state['renderer'].draw()
        visualization_area.plotly_chart(st.session_state['renderer'].fig, use_container_width=True, config=plotly_config)
    """
    import InterSim.components.side_bar as sidebar
    sidebar.draw(visualization_area)

def main():
    st.set_page_config(layout="wide")
    st.title("Interactive Covi19sim")

    visualization_area = st.empty()
    if 'renderer' in st.session_state:
        visualization_area.plotly_chart(st.session_state['renderer'].fig, use_container_width=True, config=plotly_config)

    if "log" not in st.session_state:
        st.session_state["log"] = ""

    st.code(st.session_state["log"])

    with st_stdout("code"):
        if 'sim' not in st.session_state:
            st.session_state['sim'] = get_sim(get_config())
            if 'renderer' not in st.session_state:
                st.session_state['renderer'] = visualizer(st.session_state['sim'])
                visualization_area.plotly_chart(st.session_state['renderer'].fig, use_container_width=True, config=plotly_config)

        draw_sidebar(visualization_area)
        #st.write(st.session_state["log"])


@hydra.main(config_path="../covid19sim/configs/simulation/config.yaml")
def __init__(conf: DictConfig):
    # cache config
    with open("confcache.json", "w") as outfile:
        json.dump(OmegaConf.to_container(conf), outfile)
    sys.argv = ["streamlit", "run", sys.argv[0]]
    sys.exit(stcli.main())


if __name__ == "__main__":
    if not st._is_running_with_streamlit:
        __init__()
    else:
        main()