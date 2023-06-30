from criteria import Label, accuracy, iou
import cv2
from datetime import datetime as dt
from detectors import SddCxxDRD, SddCxxLPD, SddCxxPSD
from detectors import DRDParameter, LPDParameter, PSDParameter, Parameter
from detectors import Dataset, MVTechDataset
from ext import (
    SddCxxSensitivity,
    SddCxxRectangleList,
    SddCxxRectangleLists,
    SddCxxLabelList,
    SddCxxAnomalyList,
)
import gc
from logging import getLogger, NullHandler
from logging import INFO
import numpy as np
import optuna
import os
from sklearn.model_selection import StratifiedKFold
from typing import Union
from utils import get_best_model, get_ok_imgs, save_parameter_yaml, shorten_algo_name


class OptunaOptimization:
    """
    Parameter Tuning using optuna

    Parameters
        ----------
        dataset
            Dataset object
        criteria
            criteria for auto tuning. Please see criteria.py
        timeout
            timeout for auto tuning. The unit is sec
        param_space
            list of paramerer space
        settings
            settings dict. It is originally named `settings.yml`
        show_progress_bar
            You can see the progress bar on notebook if it is True
        threshold
            Fixed threshold parameter. It will be tuned if it is None
        area
            Fixed area parameter. It will be tuned if it is None
        seed
            seed for auto tuning. You can get the same result with the same seed and dataset
    """

    def __init__(
        self,
        dataset: MVTechDataset,
        criteria: callable(list),
        param_space: list[dict],
        settings: dict,
        timeout: float,
        show_progress_bar: bool,
        callbacks: list[callable] = None,
        threshold: float = None,
        area: float = None,
        seed: int = 1,
        loglevel: int = INFO,
    ) -> None:
        self.logger = getLogger(__name__)
        self.logger.addHandler(NullHandler())
        self.logger.setLevel(loglevel)
        self.logger.propagate = True

        self.dataset = dataset
        self._dataset_validator_train()
        self._dataset_validator_test()

        self.label = dataset.labels
        self.criteria_label = [Label(i) for i in self.label]
        self.criteria = criteria
        self.show_progress_bar = show_progress_bar
        self.callbacks = callbacks
        self.settings = settings
        self.timeout = timeout
        self.seed = seed

        # criterion ---------------------Zhu
        self.accuracy_dict = {}
        self.auc_dict = {}
        self.odr_dict = {}
        self.cls_report_dict = {}
        self.confu_matrix_dict = {}
        self.balanced_metrics_dict = {}
        self.params_dict = {}

        param_space_internal: dict[Parameter] = {}
        for param in param_space:
            param_space_internal[param["name"]] = Parameter(**param)
        self.param_space = param_space_internal
        if threshold is not None:
            self.param_space["threshold"].type = "fixed"
            self.param_space["threshold"].domain = threshold
        if area is not None:
            self.param_space["area"].type = "fixed"
            self.param_space["area"].domain = area

    def _dataset_validator_train(self):
        traindatalength = set(
            (
                len(self.dataset.train_imgs_path),
                len(self.dataset.train_imgs),
            )
        )
        if len(traindatalength) != 1:
            raise ValueError("length of train data are not same")
    def _dataset_validator_test(self):
        testdatalength = set(
            (
                len(self.dataset.test_imgs_path),
                len(self.dataset.test_imgs),
                len(self.dataset.labels),
            )
        )
        if len(testdatalength) != 1:
            raise ValueError("length of test data are not same")

    def _save_best_param_model(self, best_params_dict: dict, best_trial: optuna.Trial):
        dt_now = dt.now()
        timestamp = dt_now.strftime("%Y%m%d%H%M%S")

        algname = shorten_algo_name(best_trial.user_attrs["alg"])
        savedir = os.path.join(self.settings["save_dir"], timestamp, algname)
        if not os.path.exists(savedir):
            os.makedirs(savedir)

        best_params_dict["license_path"] = best_trial.user_attrs["license_path"]
        if "onnx" in best_trial.user_attrs:
            best_params_dict["onnx"] = best_trial.user_attrs["onnx"]

        ok_imgs, ok_masks = get_ok_imgs(self.dataset)

        get_best_model(best_trial, best_params_dict, ok_imgs, ok_masks, savedir)
        imageheight, imagewidth, _ = self.dataset.imgs[0].shape

        save_parameter_yaml(
            algname,
            best_params_dict,
            timestamp,
            imagewidth,
            imageheight,
            self.dataset.masks_path,
            self.dataset.imgs_path,
            savedir,
        )

    def optimize(self) -> tuple[dict[str, Union[int, float, str]], float]:
        """
        Returns
        ----------
        params_dict: dict
            a dictionary of the parameters
        score: np.float
            score(default accuracy) of the parameters
        best_trial: optuna.Trial.trial
            optuna trial object with best trial
        """
        self.logger.info("start Optuna optimization")

        sampler = optuna.samplers.TPESampler(seed=self.seed)
        study = optuna.create_study(
            direction="maximize", sampler=sampler, study_name="hyper tune"
        )

        study.optimize(
            lambda trial: self._objective(trial),
            n_trials= self.settings["n_trials"], # zhu added
            timeout=self.timeout,
            show_progress_bar=self.show_progress_bar,
            gc_after_trial=True,
            callbacks= None if self.callbacks is None else [self.callbacks],
        )

        best_trial = study.best_trial
        best_params_dict = study.best_params

        #self._save_best_param_model(best_params_dict, best_trial)

        self.logger.info(f"Parameter tuning Finished")

        return best_params_dict, study.best_value, best_trial

    def _suggest_val(
        self, trial: optuna.Trial, param: Parameter, key: str
    ) -> Union[int, float, str]:
        match param.type:
            case "discrete":
                val = trial.suggest_int(key, param.domain[0], param.domain[1])
            case "continuous":
                val = trial.suggest_float(key, param.domain[0], param.domain[1])
            case "categorical":
                val = trial.suggest_categorical(key, param.domain)
            case "fixed":
                trial.set_user_attr(key, param.domain)
                val = param.domain
            case _:
                raise KeyError(
                    "param.type must be discrete, continuous or categorical."
                )
        return val

    def _to_annotate_img(self, bg: np.ndarray, rs: SddCxxRectangleList) -> np.ndarray:
        annotate_img = np.zeros_like(bg, dtype=np.uint8)
        color = 1
        thickness = -1
        for r in rs:
            top_left = (r.rect.x, r.rect.y)
            bottom_right = (r.rect.x + r.rect.width - 1, r.rect.y + r.rect.height - 1)
            annotate_img = cv2.rectangle(
                annotate_img, top_left, bottom_right, color, thickness
            )
        return annotate_img

    def _objective(self, trial: optuna.Trial) -> float:
        """
        objective function for optuna

        Parameters
        ----------
        trial
            optuna trial object

        Returns
        ----------
        score: float
        """
        trial_parameter = {}

        # produce a dictionary of the parameters
        # avoid patch_size less than extraction_step

        for param_name in self.param_space:
            # param_space: {'param_name': (range)}
            if "extraction_step_height" in param_name:
                extraction_step_height_domain = self.param_space[param_name]
                # store (range_height) to extraction_step_height_domain
                if (
                    extraction_step_height_domain.domain[0]
                    > trial_parameter["patch_size_height"]
                ):
                    # compare smallest, if dict's small, use dict's
                    extraction_step_height_domain.domain[0] = trial_parameter[
                        "patch_size_height"
                    ]
                trial_parameter[param_name] = self._suggest_val(
                    trial, extraction_step_height_domain, param_name
                )
                # from trial, get suggested value
                continue
            if "extraction_step_width" in param_name:
                extraction_step_width_domain = self.param_space[param_name]
                if (
                    extraction_step_width_domain.domain[0]
                    > trial_parameter["patch_size_width"]
                ):
                    extraction_step_width_domain.domain[0] = trial_parameter[
                        "patch_size_width"
                    ]
                trial_parameter[param_name] = self._suggest_val(
                    trial, extraction_step_width_domain, param_name
                )
                continue
            trial_parameter[param_name] = self._suggest_val(
                trial, self.param_space[param_name], param_name
            )

        self.model_params = {}
        self.sensitivity_params = {}
        for tr_param in trial_parameter:
            if (tr_param == "threshold") or (tr_param == "area"):
                self.sensitivity_params[tr_param] = trial_parameter[tr_param]
            elif (tr_param == "combine_weights") or (tr_param == "partial_weights"):
                self.model_params[tr_param] = SddCxxAnomalyList(trial_parameter[tr_param])
            else:
                self.model_params[tr_param] = trial_parameter[tr_param]
        self.logger.info(f"model_params: {self.model_params}")

        match trial_parameter["alg"]:
            # "alg" for algorithm, choose one of the following
            case "DictionaryReconstructDetector":
                self.detector = SddCxxDRD(DRDParameter(**self.model_params))
            case "LocalPatternDetector":
                self.detector = SddCxxLPD(LPDParameter(**self.model_params))
            case "PatchSampleDetector":
                self.detector = SddCxxPSD(PSDParameter(**self.model_params))
            case _:
                ValueError("alg is invalid")

        thresh_anomaly = self.sensitivity_params["threshold"]
        area = self.sensitivity_params["area"]

        results = []

        self.detector.fit(self.dataset.train_imgs)
        anomaly_imgs = self.detector.predict(self.dataset.test_imgs)

        scs = SddCxxSensitivity()
        scs.sen.anomaly = thresh_anomaly
        scs.sen.area = area
        rects = SddCxxRectangleLists()
        labels = SddCxxLabelList()
        anomalys = SddCxxAnomalyList()
        self.detector.interpret(
            scs=scs,
            anomaly_imgs=anomaly_imgs,
            rects=rects,
            labels=labels,
            anomalys=anomalys,
        )

        criteria_label = self.criteria_label
        # if self.criteria.__name__ == "iou":
        #     for anomaly_img, rs in zip(anomaly_imgs, rects):
        #         annotate_imgs.append(self._to_annotate_img(anomaly_img, rs))
        #     results.append(self.criteria(annotate_imgs, ground_truth))
        
        # zhu here, want high auc and low odr(over detection rate)
        if self.criteria.__name__ == "auc_odr":
            auc, odr, balanced_metrics, cls_report, confu_mat = self.criteria(anomalys, criteria_label)
            self.auc_dict[str(trial.number)] = auc # store auc
            self.odr_dict[str(trial.number)] = odr # store odr
            self.params_dict[str(trial.number)] = self.model_params # store params
            # write here, want high auc and low odr
            auc_weight = 1
            odr_weight = 1
            res = auc_weight * auc - odr_weight * odr
            results.append(res)
            # _make_cls_report
            self.cls_report_dict[str(trial.number)] = cls_report 
            # _make_confu_matrix
            self.confu_matrix_dict[str(trial.number)] = confu_mat
            # get balanced metrics
            self.balanced_metrics_dict[str(trial.number)] = balanced_metrics

        else:
            labels_tmp = []
            for label in labels:
                labels_tmp.append(Label(label))
            results.append(self.criteria(labels_tmp, criteria_label))

        gc.collect()

        del self.detector
        gc.collect()
        return results
    

