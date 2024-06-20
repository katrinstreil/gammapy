# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Built-in models in Gammapy."""
from gammapy.utils.registry import Registry
from .core import DatasetModels, Model, ModelBase, Models
from .cube import (
    FoVBackgroundModel,
    SkyModel,
    TemplateNPredModel,
    create_fermi_isotropic_diffuse_model,
)
from .IRF import EffAreaIRFModel, ERecoIRFModel, IRFModel, IRFModels
from .prior import GaussianPrior, MultiVariantePrior, Prior, UniformPrior
from .spatial import (
    ConstantFluxSpatialModel,
    ConstantSpatialModel,
    DiskSpatialModel,
    GaussianSpatialModel,
    GeneralizedGaussianSpatialModel,
    MyCustomBandModel,
    PiecewiseNorm3DModel,
    PiecewiseNormSpatialModel,
    PointSpatialModel,
    Shell2SpatialModel,
    ShellSpatialModel,
    SpatialModel,
    TemplateNDSpatialModel,
    TemplateSpatialModel,
)
from .spectral import (
    BrokenPowerLawNormSpectralModel,
    BrokenPowerLawSpectralModel,
    CompoundNormSpectralModel,
    CompoundSpectralModel,
    ConstantSpectralModel,
    EBLAbsorptionNormSpectralModel,
    ExpCutoffPowerLaw3FGLSpectralModel,
    ExpCutoffPowerLawNormSpectralModel,
    ExpCutoffPowerLawSpectralModel,
    GaussianSpectralModel,
    LogParabolaNormSpectralModel,
    LogParabolaSpectralModel,
    NaimaSpectralModel,
    PiecewiseNormPenSpectralModel,
    PiecewiseNormSpectralModel,
    PowerLaw2SpectralModel,
    PowerLawNormEffAreaSpectralModel,
    PowerLawNormOneHundredSpectralModel,
    PowerLawNormSpectralModel,
    PowerLawNornSpectralModel,
    PowerLawSpectralModel,
    ScaleSpectralModel,
    SmoothBrokenPowerLawSpectralModel,
    SpectralModel,
    SuperExpCutoffPowerLaw3FGLSpectralModel,
    SuperExpCutoffPowerLaw4FGLDR3SpectralModel,
    SuperExpCutoffPowerLaw4FGLSpectralModel,
    TemplateNDSpectralModel,
    TemplateSpectralModel,
    integrate_spectrum,
    scale_plot_flux,
)
from .spectral_cosmic_ray import create_cosmic_ray_spectral_model
from .spectral_crab import MeyerCrabSpectralModel, create_crab_spectral_model
from .temporal import (
    ConstantTemporalModel,
    ExpDecayTemporalModel,
    GaussianTemporalModel,
    GeneralizedGaussianTemporalModel,
    LightCurveTemplateTemporalModel,
    LinearTemporalModel,
    PowerLawTemporalModel,
    SineTemporalModel,
    TemplatePhaseCurveTemporalModel,
    TemporalModel,
)

__all__ = [
    "MyCustomBandModel",
    "BrokenPowerLawSpectralModel",
    "BrokenPowerLawNormSpectralModel",
    "CompoundSpectralModel",
    "CompoundNormSpectralModel",
    "ConstantFluxSpatialModel",
    "ConstantSpatialModel",
    "ConstantSpectralModel",
    "ConstantTemporalModel",
    "create_cosmic_ray_spectral_model",
    "create_crab_spectral_model",
    "create_fermi_isotropic_diffuse_model",
    "DatasetModels",
    "DiskSpatialModel",
    "EBLAbsorptionNormSpectralModel",
    "EffAreaIRFModel",
    "ERecoIRFModel",
    "ExpCutoffPowerLaw3FGLSpectralModel",
    "ExpCutoffPowerLawNormSpectralModel",
    "ExpCutoffPowerLawSpectralModel",
    "ExpDecayTemporalModel",
    "FoVBackgroundModel",
    "GaussianSpatialModel",
    "GaussianSpectralModel",
    "GaussianTemporalModel",
    "GeneralizedGaussianSpatialModel",
    "GeneralizedGaussianTemporalModel",
    "integrate_spectrum",
    "IRFModels",
    "LightCurveTemplateTemporalModel",
    "LinearTemporalModel",
    "LogParabolaNormSpectralModel",
    "LogParabolaSpectralModel",
    "MeyerCrabSpectralModel",
    "Model",
    "Models",
    "ModelBase",
    "MODEL_REGISTRY",
    "NaimaSpectralModel",
    "PiecewiseNormSpectralModel",
    "PiecewiseNormSpatialModel",
    "PiecewiseNorm3DModel",
    "PointSpatialModel",
    "PowerLaw2SpectralModel",
    "PowerLawNormSpectralModel",
    "PowerLawSpectralModel",
    "PowerLawNornSpectralModel",
    "PowerLawNormEffAreaSpectralModel",
    "PowerLawTemporalModel",
    "PiecewiseNormPenSpectralModel",
    "PowerLawNormOneHundredSpectralModel",
    "scale_plot_flux",
    "ScaleSpectralModel",
    "Shell2SpatialModel",
    "ShellSpatialModel",
    "SineTemporalModel",
    "SkyModel",
    "SmoothBrokenPowerLawSpectralModel",
    "SPATIAL_MODEL_REGISTRY",
    "SpatialModel",
    "SPECTRAL_MODEL_REGISTRY",
    "SpectralModel",
    "SuperExpCutoffPowerLaw3FGLSpectralModel",
    "SuperExpCutoffPowerLaw4FGLDR3SpectralModel",
    "SuperExpCutoffPowerLaw4FGLSpectralModel",
    "TemplatePhaseCurveTemporalModel",
    "TemplateSpatialModel",
    "TemplateSpectralModel",
    "TemplateNDSpatialModel",
    "TemplateNDSpectralModel",
    "TemplateNPredModel",
    "TEMPORAL_MODEL_REGISTRY",
    "TemporalModel",
]


SPATIAL_MODEL_REGISTRY = Registry(
    [
        MyCustomBandModel,
        ConstantSpatialModel,
        TemplateSpatialModel,
        TemplateNDSpatialModel,
        DiskSpatialModel,
        GaussianSpatialModel,
        GeneralizedGaussianSpatialModel,
        PiecewiseNormSpatialModel,
        PiecewiseNorm3DModel,
        PointSpatialModel,
        ShellSpatialModel,
        Shell2SpatialModel,
    ]
)
"""Registry of spatial model classes."""

SPECTRAL_MODEL_REGISTRY = Registry(
    [
        ConstantSpectralModel,
        CompoundSpectralModel,
        CompoundNormSpectralModel,
        PowerLawSpectralModel,
        PowerLaw2SpectralModel,
        PowerLawNormEffAreaSpectralModel,
        BrokenPowerLawNormSpectralModel,
        BrokenPowerLawSpectralModel,
        SmoothBrokenPowerLawSpectralModel,
        PiecewiseNormSpectralModel,
        PiecewiseNormPenSpectralModel,
        PowerLawNornSpectralModel,
        PowerLawNormOneHundredSpectralModel,
        ExpCutoffPowerLawSpectralModel,
        ExpCutoffPowerLaw3FGLSpectralModel,
        SuperExpCutoffPowerLaw3FGLSpectralModel,
        SuperExpCutoffPowerLaw4FGLDR3SpectralModel,
        SuperExpCutoffPowerLaw4FGLSpectralModel,
        LogParabolaSpectralModel,
        TemplateSpectralModel,
        TemplateNDSpectralModel,
        GaussianSpectralModel,
        EBLAbsorptionNormSpectralModel,
        NaimaSpectralModel,
        ScaleSpectralModel,
        PowerLawNormSpectralModel,
        LogParabolaNormSpectralModel,
        ExpCutoffPowerLawNormSpectralModel,
    ]
)
"""Registry of spectral model classes."""

TEMPORAL_MODEL_REGISTRY = Registry(
    [
        ConstantTemporalModel,
        LinearTemporalModel,
        LightCurveTemplateTemporalModel,
        ExpDecayTemporalModel,
        GaussianTemporalModel,
        GeneralizedGaussianTemporalModel,
        PowerLawTemporalModel,
        SineTemporalModel,
        TemplatePhaseCurveTemporalModel,
    ]
)
"""Registry of temporal models classes."""

PRIOR_REGISTRY = Registry([UniformPrior, GaussianPrior, MultiVariantePrior])
"""Registry of prior classes."""

IRF_REGEISTRY = Registry([EffAreaIRFModel, ERecoIRFModel, IRFModels])

MODEL_REGISTRY = Registry([SkyModel, FoVBackgroundModel, TemplateNPredModel, IRFModels])
"""Registry of model classes"""
