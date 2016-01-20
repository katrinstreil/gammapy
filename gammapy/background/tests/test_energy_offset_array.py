from ..energy_offset_array import EnergyOffsetArray
from gammapy.data import DataStore
from gammapy.data import EventList
from ...datasets import gammapy_extra
from ...utils.testing import requires_dependency, requires_data
import numpy as np
#import pylab as pt


@requires_dependency('matplotlib')
@requires_data('gammapy-extra')
def test_energy_offset_array():
    dir=str(gammapy_extra.dir)+'/datasets/hess-crab4'
    data_store = DataStore.from_dir(dir)
    Observation_Table=data_store.obs_table
    event_list_files = data_store.make_table_of_files(Observation_Table,
                                                          filetypes=['events'])
    ev_list=[]
    for (i,i_ev_file) in enumerate(event_list_files['filename']):
        ev_list.append(EventList.read(i_ev_file))
        energy=np.logspace(-1,2 , 100)
        offset=np.linspace(0, 2.5, 100)
        Array=EnergyOffsetArray(energy,offset)
        Array.fill_events(ev_list)
        g=Array.plot_image()
        g.figure.savefig("/Users/jouvin/gammapy/gammapy/background/tests/test_energy_offset_array.png")

    
