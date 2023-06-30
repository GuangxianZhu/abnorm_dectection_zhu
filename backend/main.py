from logging import INFO, DEBUG, Formatter, StreamHandler, getLogger
from criteria import accuracy, iou
from datetime import datetime as dt
import fire, os
from hypersearch import OptunaOptimization
from utils import (
    convert_min_to_sec,
    load_domain,
    load_dataset,
    load_settings,
)


def run_hypersearch(
    criteria=iou,
    timeout=1,
    seed=1,
    threshold=None,
    area=None,
    domain_path="domain.yml",
    settings_path="settings.yml",
    debug=None
):
    if debug is not None:
        loglevel = DEBUG
    else:
        loglevel = INFO
    logger.setLevel(loglevel)
    logger.info("Start HyperSearch")
    domain = load_domain(domain_path)
    settings = load_settings(settings_path)
    dataset = load_dataset(settings)

    match criteria:
        case "iou":
            criteria = iou
        case "accuracy":
            criteria = accuracy
        case _:
            ValueError("criteria must be iou or accuracy")
    pt = OptunaOptimization(
        dataset=dataset,
        criteria=criteria,
        param_space=domain[settings["algorithm"]],
        settings=settings,
        show_progress_bar=False,
        timeout=convert_min_to_sec(timeout),
        threshold=threshold,
        area=area,
        seed=seed,
        loglevel=loglevel,
    )
    _ = pt.optimize()# -> optimize->study.optimize->self._objective(trial)->param sel + detector sel + data split + train + eval.


if __name__ == "__main__":
    logger = getLogger()
    formatter = Formatter(
        "%(asctime)s %(name)s %(funcName)s [%(levelname)s]: %(message)s"
    )
    handler = StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    fire.Fire(run_hypersearch)


