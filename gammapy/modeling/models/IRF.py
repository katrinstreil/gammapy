# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Spectral models for Gammapy."""
import logging
import operator
import os
from pathlib import Path
import numpy as np
import scipy.optimize
import scipy.special
import astropy.units as u
from astropy import constants as const
from astropy.table import Table
from astropy.utils.decorators import classproperty
from astropy.visualization import quantity_support
import matplotlib.pyplot as plt
from gammapy.maps import MapAxis, RegionNDMap
from gammapy.modeling import Parameter, Parameters
from gammapy.modeling.parameter import _get_parameters_str
from gammapy.utils.integrate import trapz_loglog
from gammapy.utils.interpolation import (
    ScaledRegularGridInterpolator,
    interpolation_scale,
)
from gammapy.utils.roots import find_roots
from gammapy.utils.scripts import make_path
from .core import ModelBase

from gammapy.modeling import Covariance, Parameter, Parameters
from gammapy.modeling.covariance import copy_covariance


__all__ = ["IRFModels", "IRFModel", "ERecoIRFModel"]

class IRFModels(ModelBase):  
    """IRF model base class. like SkyModel"""
    tag = ["IRFModels"]
    _type = "irf"
        
    def __init__(
        self,
        e_reco_model=None,
        eff_area_model=None,
        name=None,
        datasets_names=None,
    ):
        self.e_reco_model = e_reco_model
        self.eff_area_model = None
        self._name = "Irfname"    
        self.datasets_names = datasets_names    
        
        super().__init__()
        
    def __call__(self, energy_axis_true, energy_axis):
        kwargs = {par.name: par.quantity for par in self.parameters}
        #kwargs = self._convert_evaluate_unit(kwargs, energy)
        return self.evaluate(energy_axis_true, energy_axis, **kwargs)
    
    @property
    def name(self):
        """Model name"""
        return f"{self.datasets_names}-irf"
    
       
    @property
    def _models(self):
        models = self.e_reco_model, self.eff_area_model
        return [model for model in models if model is not None]
    
    
    @property
    def parameters(self):
        parameters = []
        parameters.append(self.e_reco_model.parameters)
        return Parameters.from_stack(parameters)
    
    
    def _check_covariance(self):
        if not self.parameters == self._covariance.parameters:
            self._covariance = Covariance.from_stack(
                [model.covariance for model in self._models],
            )
    
    @property
    def covariance(self):
        self._check_covariance()
        for model in self._models:
            self._covariance.set_subcovariance(model.covariance)
        return self._covariance

    @covariance.setter
    def covariance(self, covariance):
        self._check_covariance()
        self._covariance.data = covariance
        for model in self._models:
            subcovar = self._covariance.get_subcovariance(model.covariance.parameters)
            model.covariance = subcovar

    def to_dict(self, full_output=False):
        """Create dict for YAML serilisation"""
        data = {}
        data["name"] = self.name
        data["type"] = self.tag[0]

        if self.datasets_names is not None:
            data["datasets_names"] = self.datasets_names

        if self.e_reco_model is not None:
            data.update(self.e_reco_model.to_dict(full_output))

        if self.eff_area_model is not None:
            data.update(self.eff_area_model.to_dict(full_output))

        return data



    @classmethod
    def from_dict(cls, data):
        """Create IRFModels from dict"""
        
        eff_area_model_data = data.get("eff_area_model")

        if eff_area_model_data is not None:
        	# do stuff
            print("no eff area model yet")
        else:
            eff_area_model = None

        e_reco_model_data = data.get("e_reco_model")
	
        if e_reco_model_data is not None:
            e_reco_model = ERecoIRFModel.from_dict({"ERecoIRFModel": e_reco_model_data})
        else:
            e_reco_model = None
        

        return cls(
            name=data["name"],
            e_reco_model=e_reco_model,
            eff_area_model=eff_area_model,
            datasets_names=data.get("datasets_names"),
        )
        
        
    def __str__(self):
        str_ = f"{self.__class__.__name__}\n\n"

        str_ += "\t{:26}: {}\n".format("Name", self.name)

        str_ += "\t{:26}: {}\n".format("Datasets names", self.datasets_names)


        if self.e_reco_model is not None:
            e_reco_model_type = self.e_reco_model.__class__.__name__
        else:
            e_reco_model_type = ""
        str_ += "\t{:26}: {}\n".format("EReco  model type", e_reco_model_type)

        if self.eff_area_model is not None:
            eff_area_model_type = self.eff_area_model.__class__.__name__
        else:
            eff_area_model_type = ""
        str_ += "\t{:26}: {}\n".format("Eff area  model type", eff_area_model_type)

        str_ += "\tParameters:\n"
        info = _get_parameters_str(self.parameters)
        lines = info.split("\n")
        str_ += "\t" + "\n\t".join(lines[:-1])

        str_ += "\n\n"
        return str_.expandtabs(tabsize=2)    
        

class IRFModel(ModelBase):
    """IRF model base class."""

    _type = "irf"
        
    def __call__(self, energy_axis_true, energy_axis):
        kwargs = {par.name: par.quantity for par in self.parameters}
        #kwargs = self._convert_evaluate_unit(kwargs, energy)
        return self.evaluate(energy_axis_true, energy_axis, **kwargs)
    
    
class ERecoIRFModel(IRFModel):
   
    bias = Parameter("bias", "0", is_penalised=True)
    resolution = Parameter("resolution", "0", is_penalised=True)

    tag = ["ERecoIRFModel", "ereco"]
    _type = "e_reco_model"
    
    
    @staticmethod
    def evaluate(energy_axis_true, energy_axis, bias, resolution):
        from gammapy.irf import EDispKernel
        gaussian = EDispKernel.from_gauss(
            energy_axis_true=energy_axis_true,
            energy_axis=energy_axis,
            sigma=(1e-12 + np.abs(resolution.value)),
            bias=bias.value,
        )
        return gaussian
