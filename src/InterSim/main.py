import hydra
import json
import sys
import streamlit as st

from omegaconf import DictConfig, OmegaConf
from streamlit import cli as stcli
from covid19sim.interactivity.controlled_run import ControlledSimulation

timesteps = {"1 second": 1.0,
             "10 seconds": 10.0,
             "1 minute": 60.0,
             "10 minutes": 600.0,
             "1 hour": 3600.0,
             "4 hours": 14400.0,
             "12 hours": 43200.0,
             "24 hours": 86400.0}

def get_sim(conf: DictConfig):
    sim = ControlledSimulation(conf)
    return sim


def get_config():
    with open("confcache.json", "r") as infile:
        return OmegaConf.create(json.load(infile))


def draw_sidebar():
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

def main():
    st.title("Interactive Covi19sim")

    visualization_area = st.empty()
    visualization_area.write(str(st.session_state['sim']) if 'sim' in st.session_state else None)

    console_area = st.empty()
    console_area.code("[CONSOLE OUTPUT HERE]")

    if 'sim' not in st.session_state:
        st.session_state['sim'] = get_sim(get_config())
        visualization_area.write(str(st.session_state['sim']))


    draw_sidebar()


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