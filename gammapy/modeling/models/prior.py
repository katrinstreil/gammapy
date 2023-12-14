# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
import numpy as np
import astropy.units as u
from gammapy.modeling import Parameter, Parameters, PriorParameter, PriorParameters
from .core import ModelBase

log = logging.getLogger(__name__)


def _build_priorparameters_from_dict(data, default_parameters):
    """Build PriorParameters object from input dict and the default prior parameter values from the PriorModel class."""
    par_data = []

    input_names = [_["name"] for _ in data]

    for par in default_parameters:
        par_dict = par.to_dict()
        try:
            index = input_names.index(par_dict["name"])
            par_dict.update(data[index])
        except ValueError:
            log.warning(
                f"PriorParameter '{par_dict['name']}' not defined in YAML file."
                f" Using default value: {par_dict['value']} {par_dict['unit']}"
            )
        par_data.append(par_dict)

    return PriorParameters.from_dict(par_data)


class Prior(ModelBase):
    _unit = ""

    def __init__(self, modelparameters, **kwargs):

        if isinstance(modelparameters, Parameter):
            self._modelparameters = Parameters([modelparameters])
        elif isinstance(modelparameters, Parameters):
            self._modelparameters = modelparameters
        else:
            raise ValueError(f"Invalid model type {modelparameters}")

        # Copy default parameters from the class to the instance
        default_parameters = self.default_parameters.copy()

        for par in default_parameters:
            value = kwargs.get(par.name, par)
            if not isinstance(value, PriorParameter):
                par.quantity = u.Quantity(value)
            else:
                par = value

            setattr(self, par.name, par)

        _weight = kwargs.get("weight", None)

        if _weight is not None:
            self._weight = _weight
        else:
            self._weight = 1

        for par in self._modelparameters:
            par.prior = self

    @property
    def modelparameters(self):
        return self._modelparameters

    @property
    def parameters(self):
        """PriorParameters (`~gammapy.modeling.PriorParameters`)"""
        return PriorParameters(
            [getattr(self, name) for name in self.default_parameters.names]
        )

    def __init_subclass__(cls, **kwargs):
        # Add priorparameters list on the model sub-class (not instances)
        cls.default_parameters = PriorParameters(
            [_ for _ in cls.__dict__.values() if isinstance(_, PriorParameter)]
        )

    @property
    def weight(self):
        return self._weight

    @weight.setter
    def weight(self, value):
        self._weight = value

    def __call__(self):
        """Call evaluate method"""
        # assuming the same unit as the PriorParameter here
        kwargs = {par.name: par.value for par in self.parameters}
        return self.weight * self.evaluate(self._modelparameters.value[0], **kwargs)

    def to_dict(self, full_output=False):
        """Create dict for YAML serialisation"""
        tag = self.tag[0] if isinstance(self.tag, list) else self.tag
        params = self.parameters.to_dict()

        if not full_output:
            for par, par_default in zip(params, self.default_parameters):
                init = par_default.to_dict()
                for item in [
                    "min",
                    "max",
                    "error",
                ]:
                    default = init[item]

                    if par[item] == default or (
                        np.isnan(par[item]) and np.isnan(default)
                    ):
                        del par[item]

        data = {
            "type": tag,
            "parameters": params,
            "weight": self.weight,
            "modelparameters": self._modelparameters,
        }

        return data

    @classmethod
    def from_dict(cls, data):
        from . import PRIOR_REGISTRY

        prior_cls = PRIOR_REGISTRY.get_cls(data["type"])
        kwargs = {}

        if data["type"] not in prior_cls.tag:
            raise ValueError(
                f"Invalid model type {data['type']} for class {cls.__name__}"
            )
        priorparameters = _build_priorparameters_from_dict(
            data["parameters"], prior_cls.default_parameters
        )
        kwargs["weight"] = data["weight"]
        kwargs["modelparameters"] = data["modelparameters"]

        return prior_cls.from_parameters(priorparameters, **kwargs)


class MultiVariantePrior(Prior):
    r"""Multi-dimensional Prior.


    Parameters
    ----------
    inv_cov : array
        Inverted Covariance Matrix
    """

    tag = ["MultiVariantePrior"]
    _type = "prior"

    def __init__(self, modelparameters, covariance_matrix, name):
        self._modelparameters = modelparameters
        self.name = name

        # check the shape of the covariance matrix
        shape = np.shape(covariance_matrix)
        if len(shape) == 2 and shape[0] == shape[1]:
            self._dimension = shape[0]
            self._covariance_matrix = covariance_matrix
        else:
            raise ValueError("Covariance matrix must be quadratic.")

        # check if model parameters is the same length as the matrix
        if len(self._modelparameters) != self._dimension:
            raise ValueError("dimension mismatch")

        for par in self._modelparameters:
            par.prior = self

        super().__init__(self._modelparameters)

    def __call__(self):
        """Call evaluate method"""
        return self.evaluate(self._modelparameters.value)

    @property
    def covariance_matrix(self):
        return self._covariance_matrix

    def evaluate(self, values):
        return np.matmul(values, np.matmul(values, self.covariance_matrix))

    # not here, but in PriorModel and test if covariance matrix is set!
    def to_dict(self, full_output=False):
        """Create dict for YAML serialisation"""
        tag = self.tag[0] if isinstance(self.tag, list) else self.tag
        # params = self.modelparameters.to_dict()
        # todo: add more information about the modelparameters
        if full_output:
            data = {
                "type": tag,
                "modelparameters": self.modelparameters.names,
                "weight": self.weight,
                "name": self.name,
            }
            if self.dimension > 1:
                data["covariance_matrix"] = self.covariance_matrix
        else:
            data = {"type": tag, "name": self.name}

        if self.type is None:
            return data
        else:
            return {self.type: data}


class GaussianPrior(Prior):
    r"""One-dimensional Gaussian Prior.


    Parameters
    ----------
    mu : float
        Mean of the Gaussian distribution
        Default is 0
    sigma : float
        Standard deviation of the Gaussian distribution.
        Default is 1.
    """

    tag = ["GaussianPrior"]
    _type = "prior"
    mu = PriorParameter(name="mu", value=0)
    sigma = PriorParameter(name="sigma", value=1)

    def evaluate(self, value, mu, sigma):
        return ((value - mu) / sigma) ** 2


class UniformPrior(Prior):
    r"""Uniform Prior.

    Returns 1. if the parameter value is in (min, max).
    0. if otherwise.

    Parameters
    ----------
    min : float
        Minimum value.
        Default is -inf.

    max : float
        Maxmimum value.
        Default is inf.

    """

    tag = ["UniformPrior"]
    _type = "prior"
    min = PriorParameter(name="min", value=-np.inf, unit="")
    max = PriorParameter(name="max", value=np.inf, unit="")

    def evaluate(self, value, min, max):
        if min < value < max:
            return 1.0
        else:
            return 0.0
