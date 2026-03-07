#!/usr/bin/env python3

import hydra
import importlib
import os
import logging
from hydra.core.hydra_config import HydraConfig
from hydra.utils import call
from omegaconf import DictConfig

from src.utils.cluster import submit
from src.utils.config import update_config_from_prev, check_cfg

log = logging.getLogger("SKATR")

@hydra.main(config_path="config", config_name="rerun", version_base=None)
def main(cfg: DictConfig):

    # read hydra config
    hcfg = HydraConfig.get()

    # determine experiment directory
    exp_dir = cfg.prev_exp_dir or hcfg.runtime.output_dir

    # resolve config if loading previous experiment
    if cfg.prev_exp_dir:
        assert os.path.exists(exp_dir)
        cfg = update_config_from_prev(cfg, hcfg, exp_dir)

    # check cfg
    check_cfg(cfg, log)

    # submit this script to a cluster
    if cfg.submit:
        submit(cfg, hcfg, exp_dir, log)
        return

    # select and run experiment
    experiment = call(cfg.experiment, cfg, exp_dir)
    experiment.run()


if __name__ == "__main__":
    main()
