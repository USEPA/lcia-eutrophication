
from collections import OrderedDict

# df of geotot, flows, and compartments:
# these are unique combinations of GeoFFis and SDMs
s_geoflowcomp_dict = 'dict FF Geotot,Flow,Comp'
s_fftot_dict = 'FF tot (as dict)'


# the dictionary
d_geotot = OrderedDict()


# keys for geotot dictionary:
emit_geofile = 'Emission geofile'
ids_emit = 'IDs of emission shapefile'
ids_recv = 'IDs of receiving shapefile'
comp_start = 'Compartment start'
comp_end = 'Compartment end'
comp_start_path = 'Compartment start from geopath'
comp_end_path = 'Compartment end from geopath'
unit_start = 'Units in'
unit_end = 'Units out'
unit_start_path = 'Units start from geopath'
unit_end_path = 'Units end from geopath'
flowable_in = 'Flowable in'
flowable_out = 'Flowable out'
flowable_next = 'Flowable next'
flowable_conversion = 'Flowable conv. factor'
matrix = 'matrix'
vector = 'vector'
df_paths = 'df paths'

list_geopaths = 'List Geopaths'
list_geosteps = 'List Geosteps'

geopaths = 'GeoPaths'
geosteps = 'GeoSteps'

ff_tot = 'FF total'
ff_path = 'FF path'
ff_i = 'FFi, individual'
sdm = 'Spatial distribution matrix'

df_steps = 'df of geostep descriptors'
matrix_shape = 'Matrix shape'
matrix_sum = 'Matrix sum'
dict_ids = 'df of geostep ids'

dict_convs = 'List unit conversion steps'
dict_flows_in = 'List flows in' # used to generate conversion steps
# factor_conv = 'Conversion factor to next geostep'

step_type = 'Type GeoStep'
step_name = 'Name GeoStep'

step_geoFFi = 'FFi'
step_SDM = 'SDMat'