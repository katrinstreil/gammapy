# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Metadata base container for Gammapy."""
import json
from typing import Optional, Union
from astropy.coordinates import Angle, EarthLocation, SkyCoord
from astropy.time import Time
from astropy.units import Quantity
import yaml
from pydantic import BaseModel, validator
from gammapy.version import version

__all__ = ["MetaData", "CreatorMetaData"]


class MetaData(BaseModel):
    """Base model for all metadata classes in Gammapy."""

    class Config:
        """Global config for all metadata."""

        extra = "allow"
        arbitrary_types_allowed = True
        validate_all = True
        validate_assignment = True

        # provides a recipe to export arbitrary types to json
        json_encoders = {
            Angle: lambda v: f"{v.value} {v.unit}",
            Quantity: lambda v: f"{v.value} {v.unit}",
            Time: lambda v: f"{v.iso}",
            EarthLocation: lambda v: f"lon : {v.lon.value} {v.lon.unit}, "
            f"lat : {v.lat.value} {v.lat.unit}, "
            f"height : {v.height.value} {v.height.unit}",
            SkyCoord: lambda v: f"lon: {v.spherical.lon.value} {v.spherical.lon.unit}, "
            f"lat: {v.spherical.lat.value} {v.spherical.lat.unit}, "
            f"frame: {v.frame.name} ",
        }

    def to_header(self):
        hdr_dict = {}
        for key, item in self.dict().items():
            hdr_dict[key.upper()] = item.__str__()
        return hdr_dict

    @classmethod
    def from_header(cls, hdr):
        kwargs = {}
        for key in cls.__fields__.keys():
            kwargs[key] = hdr.get(key.upper(), None)
        return cls(**kwargs)

    def to_yaml(self):
        """Dumps metadata content to yaml."""
        meta = json.loads(self.json())
        return yaml.dump(
            meta, sort_keys=False, indent=4, width=80, default_flow_style=False
        )


class CreatorMetaData(MetaData):
    """Metadata containing information about the object creation.

    Parameters
    ----------
    creator : str
        the software used to create the data contained in the parent object
    date : `~astropy.time.Time` or str
        the creation date
    origin : str
        the organization at the origin of the data
    """

    creator: Optional[str]
    date: Optional[Union[str, Time]]
    origin: Optional[str]

    @validator("date")
    def validate_time(cls, v):
        if v is not None:
            return Time(v)
        else:
            return v

    def to_header(self, format="gadf"):
        """Convert creator metadata to fits header.

        Parameters
        ----------
        format : str
            header format. Default is 'gadf'.

        Returns
        -------
        header : dict
            the header dictionary
        """
        if format != "gadf":
            raise ValueError(f"Creator metadata: format {format} is not supported.")

        hdr_dict = {}
        hdr_dict["CREATED"] = self.date.iso
        hdr_dict["CREATOR"] = self.creator
        hdr_dict["ORIGIN"] = self.origin

        return hdr_dict

    @classmethod
    def from_default(cls):
        """Creation metadata containing current time and Gammapy version."""
        date = Time.now()
        creator = f"Gammapy {version}"
        return cls(creator=creator, date=date)

    @classmethod
    def from_header(cls, hdr, format="gadf"):
        """Builds creator metadata from fits header.

        Parameters
        ----------
        hdr : dict
            the header dictionary
        format : str
            header format. Default is 'gadf'.
        """
        if format != "gadf":
            raise ValueError(f"Creator metadata: format {format} is not supported.")

        date = hdr.get("CREATED", None)  # note regular FITS keyword is DATE
        origin = hdr.get("ORIGIN", None)
        creator = hdr.get("CREATOR", None)
        return cls(creator=creator, date=date, origin=origin)
