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
from gammapy.utils.integrate import trapz_loglog
from gammapy.utils.interpolation import (
    ScaledRegularGridInterpolator,
    interpolation_scale,
)
from gammapy.utils.roots import find_roots
from gammapy.utils.scripts import make_path
from .core import ModelBase

class IRFModels(ModelBase):  
    """IRF model base class. like SkyModel"""

    _type = "irf"
        
    def __init__(
        self,
        e_reco_model=None,
        name=None,
        datasets_names=None,
    ):
        self.e_reco_model = e_reco_model
        self.eff_area_model = None
        self._name = "Irfname"    
        self.datasets_names = datasets_names    
        
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
        models = self.e_reco_model
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

    @staticmethod
    def evaluate(energy_axis_true, energy_axis, bias, resolution):
        from gammapy.irf import EDispKernel
        print("evaluate:", resolution, bias)
        gaussian = EDispKernel.from_gauss(
            energy_axis_true=energy_axis_true,
            energy_axis=energy_axis,
            sigma=(1e-12 + np.abs(resolution.value)),
            bias=bias.value,
        )
        return gaussian