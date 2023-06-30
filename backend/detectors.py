from abc import ABC
import cv2
from dataclasses import dataclass
import numpy as np
import os
from typing import Union

from ext import (
    SddCxx,
    SddCxxParameter,
    SddCxxSensitivity,
    SddCxxRectangleLists,
    SddCxxLabelList,
    SddCxxAnomalyList,
)

Num = Union[int, float]


@dataclass
class Parameter:
    name: str
    type: str
    domain: Union[str, Num, tuple[Num, Num]]


@dataclass
class Dataset:
    imgs: list[np.array]
    masks: list[np.array]
    ground_truth: list[np.array]
    labels: list[int]
    imgs_path: list[str]
    masks_path: list[str]

@dataclass
class MVTechDataset:
    train_imgs_path: list[str]
    train_imgs: list[np.array]
    test_imgs_path: list[str]
    test_imgs: list[np.array]
    labels: list[int]


@dataclass
class DRDParameter:
    alg: str = "DictionaryReconstructDetector"
    license_path: str = "../license/node-locked.lic"
    patch_size_height: int = 8
    patch_size_width: int = 8
    extraction_step_height: int = 4
    extraction_step_width: int = 4
    deviation_step: float = 2.0
    n_components: int = 10
    max_iter: int = 10
    n_nonzero_coefs: int = 2
    train_size: int = 10000
    b_lmc: bool = False
    b_csc: bool = False
    b_acd: bool = False
    b_ss_mean: bool = True
    b_ss_scale: bool = True


@dataclass
class LPDParameter:
    alg: str = "LocalPatternDetector"
    license_path: str = "../license/node-locked.lic"
    patch_size_height: int = 32
    patch_size_width: int = 32
    extraction_step_height: int = 16
    extraction_step_width: int = 16
    n_bins: int = 8
    radius: int = 3
    diameter: int = 9
    sigmaColor: int = 75
    sigmaSpace: int = 75
    b_lmc: bool = False
    b_fha: bool = True
    combine_method: str = "submaxmul"
    combine_weights: SddCxxAnomalyList = SddCxxAnomalyList(( 0.0, 0.0 ))
    combine_gamma: float = 2.0
    smooth_method: str = "bi"
    smooth_gamma: float = 0.5
    partial_method: str = "naive"
    partial_weights: SddCxxAnomalyList = SddCxxAnomalyList((0.0, 0.0, 0.0, 0.0, 0.0))


@dataclass
class PSDParameter:
    alg: str = "PatchSampleDetector"
    license_path: str = "../license/node-locked.lic"
    # onnx: str = "../distfiles/wide_resnet50_2.onnx"
    # names: tuple[str] = ("356", "398", "460")
    onnx: str = "../distfiles/resnet18.onnx"
    names: tuple[str] = ('140', '156', '172')
    n_features: int = 112
    ridge: float = 0.01
    r_growth: float = 0.05
    b_soe: bool = False
    b_aer: bool = False


def _load_training_imgs(imgs: list[np.array], training_imgs: list[np.array]):
    for img in imgs:
        gray: np.array = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        training_imgs.append(gray)
    return training_imgs


def _load_predict_imgs(
    imgs: list[np.array], prediction_imgs: list[np.array], anomaly_imgs: list[np.array]
) -> tuple[list[np.array], list[np.array]]:
    for img in imgs:
        gray: np.array = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        prediction_imgs.append(gray)
        anomaly_img: np.array = np.empty_like(gray, dtype=np.float64)
        anomaly_imgs.append(anomaly_img)
    return prediction_imgs, anomaly_imgs


class detector(ABC):
    def fit(self, imgs: list, masks: list = None) -> None:
        training_imgs = []
        training_imgs = _load_training_imgs(imgs, training_imgs)
        if masks is None:
            self.sc.fit(imgs=training_imgs)
        else:
            self.sc.fit(imgs=training_imgs, masks=masks)

    def predict(self, imgs: list, masks: list = None) -> list[np.array]:
        prediction_imgs: list[np.array] = []
        anomaly_imgs: list[np.array] = []
        prediction_imgs, anomaly_imgs = _load_predict_imgs(
            imgs, prediction_imgs, anomaly_imgs
        )
        if masks is None:
            self.sc.predict(imgs=prediction_imgs, anomaly_imgs=anomaly_imgs)
        else:
            self.sc.predict(
                imgs=prediction_imgs, masks=masks, anomaly_imgs=anomaly_imgs
            )
        return anomaly_imgs

    def interpret(
        self,
        scs: SddCxxSensitivity,
        anomaly_imgs: list[np.array],
        rects: SddCxxRectangleLists,
        labels: SddCxxLabelList,
        anomalys: SddCxxAnomalyList,
    ) -> None:
        self.sc.interpret(scs, anomaly_imgs, rects, labels, anomalys)

    def save(
        self,
        savedir: str = "./",
        ctxname: str = "ctx_1_1.pb",
        paramsname: str = "parameter_1_1.json",
    ) -> None:
        self.sc.save(os.path.join(savedir, ctxname))
        self.scp.save(os.path.join(savedir, paramsname))


class SddCxxDRD(detector):
    def __init__(self, params: DRDParameter) -> None:
        if not isinstance(params, DRDParameter):
            raise TypeError("SddCxxDRD accepts only DRDParameter")
        self.scp = SddCxxParameter()
        self.scp.param.alg = params.alg
        self.scp.param.license.lic = params.license_path
        self.scp.param.drd.patch_size.height = params.patch_size_height
        self.scp.param.drd.patch_size.width = params.patch_size_width
        self.scp.param.drd.extraction_step.height = params.extraction_step_height
        self.scp.param.drd.extraction_step.width = params.extraction_step_width
        self.scp.param.drd.deviation_step = params.deviation_step
        self.scp.param.drd.n_components = params.n_components
        self.scp.param.drd.max_iter = params.max_iter
        self.scp.param.drd.n_nonzero_coefs = params.n_nonzero_coefs
        self.scp.param.drd.train_size = params.train_size
        self.scp.param.drd.b_lmc = params.b_lmc
        self.scp.param.drd.b_csc = params.b_csc
        self.scp.param.drd.b_acd = params.b_acd
        self.scp.param.drd.b_ss_mean = params.b_ss_mean
        self.scp.param.drd.b_ss_scale = params.b_ss_scale

        self.sc = SddCxx(scp=self.scp)


class SddCxxLPD(detector):
    def __init__(self, params: LPDParameter) -> None:
        if not isinstance(params, LPDParameter):
            raise TypeError("SddCxxLPD accepts only LPDParameter")
        self.scp = SddCxxParameter()
        self.scp.param.alg = params.alg
        self.scp.param.license.lic = params.license_path
        self.scp.param.lpd.patch_size.height = params.patch_size_height
        self.scp.param.lpd.patch_size.width = params.patch_size_width
        self.scp.param.lpd.extraction_step.height = params.extraction_step_height
        self.scp.param.lpd.extraction_step.width = params.extraction_step_width
        self.scp.param.lpd.n_bins = params.n_bins
        self.scp.param.lpd.radius = params.radius
        self.scp.param.lpd.diameter = params.diameter
        self.scp.param.lpd.sigmaColor = params.sigmaColor
        self.scp.param.lpd.sigmaSpace = params.sigmaSpace
        self.scp.param.lpd.b_lmc = params.b_lmc
        self.scp.param.lpd.b_fha = params.b_fha
        self.scp.param.lpd.combine_method = params.combine_method
        self.scp.param.lpd.combine_weights = params.combine_weights
        self.scp.param.lpd.combine_gamma = params.combine_gamma
        self.scp.param.lpd.smooth_method = params.smooth_method
        self.scp.param.lpd.smooth_gamma = params.smooth_gamma
        self.scp.param.lpd.partial_method = params.partial_method
        self.scp.param.lpd.partial_weights = params.partial_weights

        self.sc = SddCxx(scp=self.scp)


class SddCxxPSD(detector):
    def __init__(self, params: PSDParameter) -> None:
        if not isinstance(params, PSDParameter):
            raise TypeError("SddCxxPSD accepts only PSDParameter")
        self.scp = SddCxxParameter()
        self.scp.param.alg = params.alg
        self.scp.param.license.lic = params.license_path
        self.scp.param.psd.onnx = params.onnx
        self.scp.param.psd.names = params.names
        self.scp.param.psd.n_features = params.n_features
        self.scp.param.psd.ridge = params.ridge
        self.scp.param.psd.r_growth = params.r_growth
        self.scp.param.psd.b_soe = params.b_soe
        self.scp.param.psd.b_aer = params.b_aer

        self.sc = SddCxx(scp=self.scp)

    def fit(self, imgs: list[np.array], masks: list[np.array] = None) -> None:
        self.sc.fit(color_imgs=imgs)

    def predict(
        self, imgs: list[np.array], masks: list[np.array] = None
    ) -> list[np.array]:
        anomaly_imgs: list[np.array] = []
        for img in imgs:
            gray: np.array = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            anomaly_img: np.array = np.empty_like(gray, dtype=np.float64)
            anomaly_imgs.append(anomaly_img)

        self.sc.predict(color_imgs=imgs, anomaly_imgs=anomaly_imgs)

        return anomaly_imgs
