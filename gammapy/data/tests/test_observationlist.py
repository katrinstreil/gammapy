# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from astropy.tests.helper import pytest, assert_quantity_allclose
from astropy.coordinates import Angle
from astropy.coordinates import SkyCoord
from ...datasets import gammapy_extra
from ...utils.testing import requires_data, requires_dependency
from ...utils.energy import EnergyBounds, Energy
from ...data import DataStore, ObservationList


@requires_dependency('scipy')
@requires_data('gammapy-extra')
def test_make_psftable():
    center = SkyCoord(83.63, 22.01, unit='deg')
    store = gammapy_extra.filename("datasets/hess-crab4-hd-hap-prod2")
    data_store = DataStore.from_dir(store)
    obs1 = data_store.obs(23523)
    obs2 = data_store.obs(23526)
    energy = EnergyBounds.equal_log_spacing(1, 10, 100, "TeV")
    energy_band = Energy([energy[0].value, energy[-1].value], energy.unit)
    psf1 = obs1.make_psf(source_position=center, energy=energy, theta=None)
    psf2 = obs2.make_psf(source_position=center, energy=energy, theta=None)
    psf1_int = psf1.table_psf_in_energy_band(energy_band, spectral_index=2.3)
    psf2_int = psf2.table_psf_in_energy_band(energy_band, spectral_index=2.3)
    obslist = ObservationList([obs1, obs2])
    psf_tot = obslist.make_psf(source_position=center, energy=energy)
    psf_tot_int = psf_tot.table_psf_in_energy_band(energy_band, spectral_index=2.3)
    # Check that the mean PSF is consistent with the individual PSFs
    # (in this case the R68 of the mean PSF is in between the R68 of the individual PSFs)
    assert_quantity_allclose(psf1_int.containment_radius(0.68), Angle(0.06012802199573515, 'deg'))
    assert_quantity_allclose(psf2_int.containment_radius(0.68), Angle(0.05637627703540551, 'deg'))
    assert_quantity_allclose(psf_tot_int.containment_radius(0.68), Angle(0.05818540290833362, 'deg'))
