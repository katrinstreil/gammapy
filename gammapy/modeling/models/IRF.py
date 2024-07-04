# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Spectral models for Gammapy."""
import numpy as np
from scipy.interpolate import interp2d
import astropy.units as u
from gammapy.maps import Map
from gammapy.modeling import Covariance, Parameter, Parameters
from gammapy.modeling.models.spectral import PowerLawNormPenSpectralModel
from gammapy.modeling.parameter import _get_parameters_str
from .core import ModelBase

__all__ = ["IRFModels", "IRFModel", "ERecoIRFModel"]


class IRFModels(ModelBase):
    """IRF model base class. like SkyModel"""

    tag = ["IRFModels"]
    _type = "irf"

    def __init__(
        self,
        e_reco_model=None,
        eff_area_model=None,
        psf_model=None,
        datasets_names=None,
    ):
        self.e_reco_model = e_reco_model
        self.eff_area_model = eff_area_model
        self.psf_model = psf_model
        self.datasets_names = datasets_names

        super().__init__()

    @property
    def name(self):
        """Model name"""
        return f"{self.datasets_names}-irf"

    @property
    def _models(self):
        models = self.e_reco_model, self.eff_area_model, self.psf_model
        return [model for model in models if model is not None]

    @property
    def parameters(self):
        parameters = []
        for m in self._models:
            parameters.append(m.parameters)
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

        if self.psf_model is not None:
            data.update(self.psf_model.to_dict(full_output))

        return data

    @classmethod
    def from_dict(cls, data):
        """Create IRFModels from dict"""

        eff_area_model_data = data.get("eff_area_model")

        if eff_area_model_data is not None:
            eff_area_model = EffAreaIRFModel.from_dict(eff_area_model_data)
        else:
            eff_area_model = None

        e_reco_model_data = data.get("e_reco_model")

        if e_reco_model_data is not None:
            e_reco_model = ERecoIRFModel.from_dict({"ERecoIRFModel": e_reco_model_data})
        else:
            e_reco_model = None

        # psf_model_model_data = data.get("psf_model_model")

        # if psf_model_model_data is not None:
        #    psf_model_model = ERecoIRFModel.from_dict(
        #        {"PSFIRFModel": psf_model_model_data}
        #    )
        # else:
        #    psf_model_model = None

        return cls(
            # name=data["name"],
            e_reco_model=e_reco_model,
            eff_area_model=eff_area_model,
            # psf_model=psf_model,
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

        if self.psf_model is not None:
            psf_model_type = self.psf_model.__class__.__name__
        else:
            psf_model_type = ""
        str_ += "\t{:26}: {}\n".format("PSF model type", psf_model_type)

        str_ += "\tParameters:\n"
        info = _get_parameters_str(self.parameters)
        lines = info.split("\n")
        str_ += "\t" + "\n\t".join(lines[:-1])

        str_ += "\n\n"
        return str_.expandtabs(tabsize=2)


class IRFModel(ModelBase):
    """IRF model base class."""

    _type = "irf"

    def __call__(self, energy_axis):
        kwargs = {par.name: par.quantity for par in self.parameters}
        # kwargs = self._convert_evaluate_unit(kwargs, energy)
        return self.evaluate(energy_axis, **kwargs)


class ERecoIRFModel(IRFModel):
    bias = Parameter("bias", "0", is_penalised=True)
    resolution = Parameter("resolution", "0", is_penalised=True)

    tag = ["ERecoIRFModel", "ereco"]
    _type = "e_reco_model"

    @staticmethod
    def evaluate(energy_axis, bias, resolution):
        from gammapy.irf import EDispKernel

        gaussian = EDispKernel.from_gauss(
            energy_axis_true=energy_axis.copy(name="energy_true"),
            energy_axis=energy_axis,
            sigma=(1e-12 + np.abs(resolution.value)),
            bias=bias.value,
        )
        return gaussian


class PSFIRFModel(IRFModel):
    sigma_psf = Parameter("sigma_psf", "1e-3", unit="deg", is_penalised=True)

    tag = ["PSFIRFModel", "psf"]
    _type = "psf_model"

    @staticmethod
    def evaluate(geom, sigma_psf):
        from gammapy.irf import PSFMap

        print("compute gaussian for  sigma 1e-3 + ", sigma_psf)
        energy_axis_true = geom.axes["energy_true"]
        rad_axis = geom.axes["rad"]

        psf_map_g = PSFMap.from_gauss(
            energy_axis_true=energy_axis_true,
            sigma=1e-3 * u.deg + sigma_psf * u.deg,
            geom=geom.to_image(),
            rad_axis=rad_axis,
        )

        return psf_map_g


class EffAreaIRFModel(ModelBase):
    tag = ["EffAreaIRFModel", "effarea"]
    _type = "eff_area_model"

    def __init__(self, spectral_model=None):
        if spectral_model is None:
            spectral_model = PowerLawNormPenSpectralModel()

        if not spectral_model.is_norm_spectral_model:
            raise ValueError("A norm spectral model is required.")

        self._spectral_model = spectral_model
        super().__init__()

    @property
    def parameters(self):
        """Model parameters"""
        return self.spectral_model.parameters

    @property
    def spectral_model(self):
        """Spectral norm model"""
        return self._spectral_model

    def evaluate_geom(self, geom):
        """Evaluate map"""
        coords = geom.get_coord(sparse=True)
        return self.evaluate(energy=coords["energy_true"])

    def evaluate(self, energy):
        """Evaluate model"""
        return self.spectral_model(energy)

    def to_dict(self, full_output=False):
        data = {}
        data["type"] = self.tag[0]
        data["spectral"] = self.spectral_model.to_dict(full_output=full_output)[
            "spectral"
        ]
        return {"eff_area_model": data}

    @classmethod
    def from_dict(cls, data):
        """Create model from dict

        Parameters
        ----------
        data : dict
            Data dictionary
        """
        from gammapy.modeling.models import SPECTRAL_MODEL_REGISTRY

        spectral_data = data.get("spectral")
        if spectral_data is not None:
            model_class = SPECTRAL_MODEL_REGISTRY.get_cls(spectral_data["type"])
            spectral_model = model_class.from_dict({"spectral": spectral_data})
        else:
            spectral_model = None

        return cls(
            spectral_model=spectral_model,
        )


class NuisanceBackgroundModel(ModelBase):
    tag = ["NuisanceBackgroundModel", "nui_bkg"]
    _type = "nui_bkg_area_model"

    def __init__(self, N_parameters=None, nuisance_mask=None, geom_bkg=None):
        self.N_parameters = N_parameters
        self.nuisance_mask = nuisance_mask
        self.geom_bkg = geom_bkg

        if self.nuisance_mask is not None:
            # for now only rectangular shapes
            shape_N = int(np.shape(self.nuisance_mask)[1])
            fac = int(self.geom_bkg.data_shape[1] / shape_N)
            self.geom_down = self.geom_bkg.downsample(fac)

        super().__init__()

    @property
    def parameters(self):
        """Model parameters"""
        return self.N_parameters

    # @staticmethod
    def evaluate_geom(self, geom):
        if geom == self.geom_bkg:
            return 1 + self.N_map().data
        else:
            print("Not the same geom as bkg")

    def N_map(self):
        """Map of the Nuisance parameters

        Helper function to evaulate the nuisance parameters in the correct geometry.

        Returns
        -------
        npred_background : `Map`
            Predicted counts from the nuisance parameters.
        """
        if self.nuisance_mask is not None:
            N_map = Map.from_geom(self.geom_down)
            N_map.data[np.where(self.nuisance_mask == True)] = self.N_parameters.value
            fac = int(self.geom_bkg.data_shape[1] / self.geom_down.data_shape[1])
            N_map_up = Map.from_geom(self.geom_bkg)

            x = self.geom_down.to_image().get_idx()[0][0] * fac + fac / 2
            y = self.geom_down.to_image().get_idx()[0][0] * fac + fac / 2
            x_new = self.geom_bkg.to_image().get_idx()[0][0]
            y_new = self.geom_bkg.to_image().get_idx()[0][0]

            for e, z_e in enumerate(N_map.data):
                f = interp2d(
                    x=x, y=y, z=z_e, kind="cubic", fill_value=None, bounds_error=False
                )
                N_map_up.data[e, :, :] = f(x_new, y_new).reshape(
                    np.shape(N_map_up.data[0, :, :])
                )

        return N_map_up

    def from_dict(self):
        print("test")
