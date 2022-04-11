import hydra
from omegaconf import DictConfig

import covid19sim.interactivity.controlled_run as cr

@hydra.main(config_path="../configs/simulation/config.yaml")
def main(conf: DictConfig):
    sim = cr.ControlledSimulation(conf)
    step_size = 3600
    while sim.env.ts_now < sim.end_time:
        current_time = sim.env.timestamp
        sim.env.run(until=sim.env.ts_now + step_size)
    sim.end_sim()

if __name__ == "__main__":
    main()
