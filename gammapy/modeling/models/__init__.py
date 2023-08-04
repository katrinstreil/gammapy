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
from .IRF import (  # IRFModel,
    EffAreaIRFModel,
    ERecoIRFModel,
    IRFModels,
    NuisanceBackgroundModel,
)
from .spatial import (
    ConstantFluxSpatialModel,
    ConstantSpatialModel,
    DiskSpatialModel,
    GaussianSpatialModel,
    GeneralizedGaussianSpatialModel,
    PointSpatialModel,
    Shell2SpatialModel,
    ShellSpatialModel,
    SpatialModel,
    TemplateSpatialModel,
)
from .spectral import (  # PowerLawNormNuisanceSpectralModel,
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
    PiecewiseNormSpectralModel,
    PowerLaw2SpectralModel,
    PowerLawNormPenSpectralModel,
    PowerLawNormSpectralModel,
    PowerLawNornSpectralModel,
    PowerLawSpectralModel,
    ScaleNuisanceSpectralModel,
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
    "BrokenPowerLawSpectralModel",
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
    "IRFModel",
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
    "NuisanceBackgroundModel",
    "PiecewiseNormSpectralModel",
    "PointSpatialModel",
    "PowerLaw2SpectralModel",
    "PowerLawNormSpectralModel",
    "PowerLawNornSpectralModel",
    "PowerLawNormPenSpectralModel",
    # "PowerLawNormNuisanceSpectralModel",
    # "PowerLawNormNuisanceESpectralModel",
    "PowerLawSpectralModel",
    "PowerLawSpectralModel",
    "PowerLawTemporalModel",
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
    "TemplateNDSpectralModel",
    "TemplateNPredModel",
    "TEMPORAL_MODEL_REGISTRY",
    "TemporalModel",
]


SPATIAL_MODEL_REGISTRY = Registry(
    [
        ConstantSpatialModel,
        TemplateSpatialModel,
        DiskSpatialModel,
        GaussianSpatialModel,
        GeneralizedGaussianSpatialModel,
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
        BrokenPowerLawSpectralModel,
        SmoothBrokenPowerLawSpectralModel,
        PiecewiseNormSpectralModel,
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
        ScaleNuisanceSpectralModel,
        PowerLawNormSpectralModel,
        PowerLawNormPenSpectralModel,
        PowerLawNornSpectralModel,
        # PowerLawNormNuisanceSpectralModel,
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

IRF_MODEL_REGISTRY = Registry([IRFModels, ERecoIRFModel, NuisanceBackgroundModel])
"""Registry of IRF models classes."""


MODEL_REGISTRY = Registry([SkyModel, FoVBackgroundModel, TemplateNPredModel, IRFModels])
"""Registry of model classes"""
