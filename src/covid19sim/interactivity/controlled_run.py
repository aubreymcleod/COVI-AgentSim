"""
Acts as a secondary entry point, where a secondary entry point, where a separate library can control the simulation.
"""
import datetime
import logging
import os
import time
import typing
from pathlib import Path

import numpy as np
from omegaconf import DictConfig

from covid19sim.locations.city import City
from covid19sim.utils.env import Env
from covid19sim.utils.constants import SECONDS_PER_DAY, SECONDS_PER_HOUR
from covid19sim.utils.mobility_planner import ACTIVITIES
from covid19sim.log.console_logger import ConsoleLogger
from covid19sim.inference.server_utils import DataCollectionServer
from covid19sim.utils.utils import dump_conf, dump_tracker_data, extract_tracker_data, parse_configuration, log


def _get_intervention_string(conf):
    """
    Consolidates all the parameters to one single string.

    Args:
        conf (dict): yaml configuration of the experiment

    Returns:
        (str): a string to identify type of intervention being run

    Raises:
        (ValueError): if RISK_MODEL is unknown
    """
    if conf['RISK_MODEL'] == "":
        type_of_run = "UNMITIGATED"
        if conf['INTERPOLATE_CONTACTS_USING_LOCKDOWN_CONTACTS']:
            type_of_run = "LOCKDOWN"

        if conf['N_BEHAVIOR_LEVELS'] > 2:
            type_of_run = "POST-LOCKDOWN NO TRACING"

        return type_of_run

    risk_model = conf['RISK_MODEL']
    n_behavior_levels = conf['N_BEHAVIOR_LEVELS']
    hhld_behavior = conf['MAKE_HOUSEHOLD_BEHAVE_SAME_AS_MAX_RISK_RESIDENT']
    type_of_run = f"{risk_model} | HHLD_BEHAVIOR_SAME_AS_MAX_RISK_RESIDENT: {hhld_behavior} | N_BEHAVIOR_LEVELS:{n_behavior_levels} |"
    if risk_model == "digital":
        type_of_run += f" N_LEVELS_USED: 2 (1st and last) |"
        type_of_run += f" TRACING_ORDER:{conf['TRACING_ORDER']} |"
        type_of_run += f" TRACE_SYMPTOMS: {conf['TRACE_SYMPTOMS']} |"
        type_of_run += f" INTERPOLATE_USING_LOCKDOWN_CONTACTS:{conf['INTERPOLATE_CONTACTS_USING_LOCKDOWN_CONTACTS']} |"
        type_of_run += f" MODIFY_BEHAVIOR: {conf['SHOULD_MODIFY_BEHAVIOR']}"
        return type_of_run

    if risk_model == "transformer":
        type_of_run += f" USE_ORACLE: {conf['USE_ORACLE']}"
        type_of_run += f" N_LEVELS_USED: {n_behavior_levels} |"
        type_of_run += f" INTERPOLATE_USING_LOCKDOWN_CONTACTS:{conf['INTERPOLATE_CONTACTS_USING_LOCKDOWN_CONTACTS']} |"
        type_of_run += f" REC_LEVEL_THRESHOLDS: {conf['REC_LEVEL_THRESHOLDS']} |"
        type_of_run += f" MAX_RISK_LEVEL: {conf['MAX_RISK_LEVEL']} |"
        type_of_run += f" MODIFY_BEHAVIOR: {conf['SHOULD_MODIFY_BEHAVIOR']} "
        type_of_run += f"\n RISK_MAPPING: {conf['RISK_MAPPING']}"
        return type_of_run

    if risk_model in ['heuristicv1', 'heuristicv2', 'heuristicv3', 'heuristicv4']:
        type_of_run += f" N_LEVELS_USED: {n_behavior_levels} |"
        type_of_run += f" INTERPOLATE_USING_LOCKDOWN_CONTACTS:{conf['INTERPOLATE_CONTACTS_USING_LOCKDOWN_CONTACTS']} |"
        type_of_run += f" MAX_RISK_LEVEL: {conf['MAX_RISK_LEVEL']} |"
        type_of_run += f" MODIFY_BEHAVIOR: {conf['SHOULD_MODIFY_BEHAVIOR']}"
        return type_of_run

    raise ValueError(f"Unknown risk model:{risk_model}")


class ControlledSimulation:
    def __init__(self, config,
                 n_people: int = 1000,
                 init_fraction_sick: float = 0.01,
                 start_time: datetime.datetime = datetime.datetime(2020, 2, 28, 0, 0),
                 simulation_days: int = 30,
                 outfile: typing.Optional[typing.AnyStr] = None,
                 out_chunk_size: typing.Optional[int] = None,
                 seed: int = 0,
                 logfile: str = None):
        self.ACTIVITIES = ACTIVITIES
        self.config = config
        self.logfile = None
        self.collection_server = None
        self.env = None
        self.city = None
        self.end_time = None

        self._build_config()
        self._build_env(n_people=n_people,
                        init_fraction_sick=init_fraction_sick,
                        start_time=start_time,
                        simulation_days=simulation_days,
                        outfile=outfile,
                        out_chunk_size=out_chunk_size,
                        seed=seed,
                        logfile=logfile)


    def _build_config(self):
        """
        Setup environment/config to run a simulation
        """
        # -------------------------------------------------
        # -----  Load the experimental configuration  -----
        # -------------------------------------------------
        self.config = parse_configuration(self.config)

        # -------------------------------------
        # -----  Create Output Directory  -----
        # -------------------------------------
        if self.config["outdir"] is None:
            self.config["outdir"] = str(Path(__file__) / "output")

        timenow = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.config["outdir"] = "{}/sim_v2_people-{}_days-{}_init-{}_uptake-{}_seed-{}_{}_{}".format(
            self.config["outdir"],
            self.config["n_people"],
            self.config["simulation_days"],
            self.config["init_fraction_sick"],
            self.config["APP_UPTAKE"],
            self.config["seed"],
            timenow,
            str(time.time_ns())[-6:])

        if Path(self.config["outdir"]).exists():
            out_path = Path(self.config["outdir"])
            out_idx = 1
            while (out_path.parent / (out_path.name + f"_{out_idx}")).exists():
                out_idx += 1
            self.config["outdir"] = str(out_path.parent / (out_path.name + f"_{out_idx}"))

        os.makedirs(self.config["outdir"])
        self.logfile = f"{self.config['outdir']}/log_{timenow}.txt"
        outfile = os.path.join(self.config["outdir"], "data")

        # ---------------------------------
        # -----  Filter-Out Warnings  -----
        # ---------------------------------
        import warnings
        # warnings.filterwarnings("ignore")

        # ----------------------------
        # ----- Init Simulation  -----
        # ----------------------------
        # correctness of configuration file
        assert not self.config['RISK_MODEL'] != "" or self.config['INTERVENTION_DAY'] >= 0, "risk model is given, but no intervnetion day specified"
        assert self.config['N_BEHAVIOR_LEVELS'] >= 2, "At least 2 behavior levels are required to model behavior changes"
        if self.config['TRACE_SYMPTOMS']:
            warnings.warn("TRACE_SYMPTOMS: True hasn't been implemented. It will have no affect.")

        log(f"RISK_MODEL = {self.config['RISK_MODEL']}", self.logfile)
        log(f"INTERVENTION_DAY = {self.config['INTERVENTION_DAY']}", self.logfile)
        log(f"seed: {self.config['seed']}", self.logfile)

        # complete decsription of intervention
        type_of_run = _get_intervention_string(self.config)
        self.config['INTERVENTION'] = type_of_run
        log(f"Type of run: {type_of_run}", self.logfile)
        if self.config['COLLECT_TRAINING_DATA']:
            data_output_path = os.path.join(self.config["outdir"], "train.zarr")
            self.collection_server = DataCollectionServer(
                data_output_path=data_output_path,
                config_backup=self.config,
                human_count=self.config['n_people'],
                simulation_days=self.config['simulation_days'])
            self.collection_server.start()
        else:
            self.collection_server = None

        self.config["outfile"] = outfile


    def _build_env(self, n_people: int = 1000,
                init_fraction_sick: float = 0.01,
                start_time: datetime.datetime = datetime.datetime(2020, 2, 28, 0, 0),
                simulation_days: int = 30,
                outfile: typing.Optional[typing.AnyStr] = None,
                out_chunk_size: typing.Optional[int] = None,
                seed: int = 0,
                logfile: str = None):
        if self.config is None:
            self.config = {}

        self.config["n_people"] = n_people
        self.config["init_fraction_sick"] = init_fraction_sick
        self.config["start_time"] = start_time
        self.config["simulation_days"] = simulation_days
        self.config["outfile"] = outfile
        self.config["out_chunk_size"] = out_chunk_size
        self.config["seed"] = seed
        self.config['logfile'] = logfile

        # set days and mixing constants
        self.config['_MEAN_DAILY_UNKNOWN_CONTACTS'] = self.config['MEAN_DAILY_UNKNOWN_CONTACTS']
        self.config['_ENVIRONMENTAL_INFECTION_KNOB'] = self.config['ENVIRONMENTAL_INFECTION_KNOB']
        self.config['_CURRENT_PREFERENTIAL_ATTACHMENT_FACTOR'] = self.config['BEGIN_PREFERENTIAL_ATTACHMENT_FACTOR']
        start_time_offset_days = self.config['COVID_START_DAY']
        intervention_start_days = self.config['INTERVENTION_DAY']

        # start of COVID spread
        self.config['COVID_SPREAD_START_TIME'] = start_time

        # start of intervention
        self.config['INTERVENTION_START_TIME'] = None
        if intervention_start_days >= 0:
            self.config['INTERVENTION_START_TIME'] = start_time + datetime.timedelta(days=intervention_start_days)

        # start of simulation without COVID
        start_time -= datetime.timedelta(days=start_time_offset_days)
        self.config['SIMULATION_START_TIME'] = str(start_time)

        # adjust the simulation days
        self.config['simulation_days'] += self.config['COVID_START_DAY']
        simulation_days = self.config['simulation_days']

        console_logger = ConsoleLogger(frequency=SECONDS_PER_DAY, logfile=logfile, conf=self.config)
        logging.root.setLevel(getattr(logging, self.config["LOGGING_LEVEL"].upper()))

        rng = np.random.RandomState(seed)
        self.env = Env(start_time)
        city_x_range = (0, 1000)
        city_y_range = (0, 1000)
        self.city = City(
            self.env, n_people, init_fraction_sick, rng, city_x_range, city_y_range, self.config, logfile
        )

        # we might need to reset the state of the clusters held in shared memory (server or not)
        if self.config.get("RESET_INFERENCE_SERVER", False):
            if self.config.get("USE_INFERENCE_SERVER"):
                inference_frontend_address = self.config.get("INFERENCE_SERVER_ADDRESS", None)
                print("requesting cluster reset from inference server...")
                from covid19sim.inference.server_utils import InferenceClient

                temporary_client = InferenceClient(
                    server_address=inference_frontend_address
                )
                temporary_client.request_reset()
            else:
                from covid19sim.inference.heavy_jobs import DummyMemManager

                DummyMemManager.global_cluster_map = {}

        # Initiate city process, which runs every hour
        self.env.process(self.city.run(SECONDS_PER_HOUR, outfile))

        # initiate humans
        for human in self.city.humans:
            self.env.process(human.run())

        self.env.process(console_logger.run(self.env, city=self.city))
        self.end_time = self.env.ts_initial + simulation_days * SECONDS_PER_DAY


    def step(self, timestep):
        """
        run the simulation, one step at a time.

        Args:
            env: the simulation environment
            timestep: the timestep which this simulation will run until.
        """
        # Run simulation until termination
        self.env.run(until=timestep)

    def end_sim(self):
        # write the full configuration file along with git commit hash
        dump_conf(self.city.conf, "{}/full_configuration.yaml".format(self.city.conf["outdir"]))

        # log the simulation statistics
        self.city.tracker.write_metrics()

        # (baseball-cards) write full simulation data
        if hasattr(self.city, "tracker") and \
                hasattr(self.city.tracker, "collection_server") and \
                isinstance(self.city.tracker.collection_server, DataCollectionServer) and \
                self.city.tracker.collection_server is not None:
            self.city.tracker.collection_server.stop_gracefully()
            self.city.tracker.collection_server.join()

        # if COLLECT_TRAINING_DATA is true
        if not self.config["tune"]:
            # ----------------------------------------------
            # -----  Not Tune: Collect Training Data   -----
            # ----------------------------------------------
            # write values to train with
            train_priors = os.path.join(f"{self.config['outdir']}/train_priors.pkl")
            self.city.tracker.write_for_training(self.city.humans, train_priors, self.config)

            timenow = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            log("Dumping Tracker Data in {}".format(self.config["outdir"]), self.logfile)

            Path(self.config["outdir"]).mkdir(parents=True, exist_ok=True)
            filename = f"tracker_data_n_{self.config['n_people']}_seed_{self.config['seed']}_{timenow}.pkl"
            data = extract_tracker_data(self.city.tracker, self.config)
            dump_tracker_data(data, self.config["outdir"], filename)
        else:
            # ------------------------------------------------------
            # -----     Tune: Write logs And Tacker Data       -----
            # ------------------------------------------------------
            timenow = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            log("Dumping Tracker Data in {}".format(self.config["outdir"]), self.logfile)

            Path(self.config["outdir"]).mkdir(parents=True, exist_ok=True)
            filename = f"tracker_data_n_{self.config['n_people']}_seed_{self.config['seed']}_{timenow}.pkl"
            data = extract_tracker_data(self.city.tracker, self.config)
            dump_tracker_data(data, self.config["outdir"], filename)
        # Shutdown the data collection server if one's running
        if self.collection_server is not None:
            self.collection_server.stop_gracefully()
            self.collection_server.join()
            # Remove the IPCs if they were stored somewhere custom
            if os.environ.get("COVID19SIM_IPC_PATH", None) is not None:
                print("<<<<<<<< Cleaning Up >>>>>>>>")
                for file in Path(os.environ.get("COVID19SIM_IPC_PATH")).iterdir():
                    if file.name.endswith(".ipc"):
                        print(f"Removing {str(file)}...")
                        os.remove(str(file))