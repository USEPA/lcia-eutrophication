
import pandas as pd
import calc.setup.config as cfg

df_output = pd.DataFrame(data=None)



# consistent order is
# target, source, weight1, weight2

idx_target = 0
idx_source = 1
idx_weight1 = 2
idx_weight2 = 3

# NOTE: these names are embedded in the pre-calculated intersect files...
# so changing them will generate errors, as the script will open the files
# and try to grab columns based on these names.

shp_id = 'id'
shp_value = 'val'
shp_area = 'area'
shp_isect_area = 'isect' + shp_area

shp_target = 'target'
shp_source = 'source'
shp_weight1 = 'weight1'
shp_weight2 = 'weight2'

# the 'target, source, weight1, weight2 order is used in
# aggregation.py

# TODO: these lists could be OrderedDicts, to make accessing easier.
# But note that the order is indeed important

shp_list_names = [shp_target, shp_source, shp_weight1, shp_weight2]

shp_list_ids = [shp_id + '_' + n  for n in shp_list_names]
shp_list_areas = [shp_area + '_' + n  for n in shp_list_names]
shp_list_values = [shp_value + '_' + n if n != '' else '' 
                 for n in ['', shp_source, shp_weight1, shp_weight2]]


# for dictionaries of columns (to pass to aggregation functions)
key_id = shp_id
key_val = shp_value
key_name = 'name'  # weight1 or weight2
key_area = shp_area
key_type = 'type'  # string intensive or extensive
key_scale = 'scaling'  # int or float scaling factor

str_extensive = 'extensive'  # string to match if extensive
str_intensive = 'intensive'

str_aggcol_flag_area_weight = 'Flag Area Weighting Added'
str_aggcol_avg_value = 'Average Target Value'


# define columns to keep in the aggregation output:



list_calc_cols_for_index = [cfg.s_calc_ic,
                            cfg.s_calc_flowable,
                            cfg.s_calc_unit_in,
                            cfg.s_calc_geotot,
                            cfg.s_calc_factordirect,
                            cfg.s_calc_comp_emit,
                            cfg.s_calc_comp_recv,
                            cfg.s_calc_aggreg,
                            cfg.s_calc_projection,
                            cfg.s_calc_sector,
                            cfg.s_calc_data1,
                            cfg.s_calc_data2,
                            cfg.s_calc_type,
                            cfg.s_calc_unit_end]

list_agg_cols_keep = [shp_list_ids[idx_target],
                      str_aggcol_flag_area_weight,
                      str_aggcol_avg_value]

str_outputcol_unitfinal = 'Unit Final'
str_outputcol_is_normalized = 'Is Normalized'
str_outputcol_normref_descrip = 'Normalization Ref'
str_outputcol_is_normref = 'Is Normaliation Ref'
str_outputcol_nametarget = 'Target Name'

list_output_cols = []
# will be modified, depending on whether we normalize
# [str_outputcol_is_normalized,
#                    str_outputcol_normref_descrip,
#                    str_outputcol_unitfinal,
#                     str_outputcol_nametarget] + list_agg_cols_keep



# normalization
# global variable for normaization:
bool_normalize_results = False

# alternate aggregations:

s_alttype_merge = 'merge'
s_alttype_sum = 'sum'
s_alttype_exist = 'existing'

s_altkey_type = 'type'
s_altkey_data = 'dataframe'
s_altkey_altid = 'existing alternate id' # for s_alt_exist (i.e., when the id is already in the datafrome)
s_altkey_newname = 'new target name' # will become the column header, and thus will become an index in the big result df

s_altagg_country = 'Countries'  # matches the gis file, and thus the text in the Target Aggregation column
s_altagg_uscounty = 'US_County' # as above


# for aggregating countries to continents:
d_alt_cont = {s_altkey_type: s_alttype_exist,
              # s_altkey_data: df_alt,
              s_altkey_altid: 'CONTINENT',  # column id in the existing df, from which we get col_values
              s_altkey_newname: 'Continent'}  # new column display name

# for aggregating via summing countries:
d_alt_world = {s_altkey_type: s_alttype_sum,
               s_altkey_altid: 'WORLD',  # column id added to df
               s_altkey_newname: 'World'}  # display name

# for aggregating US counties to states:
d_alt_usstates = {s_altkey_type: s_alttype_exist,
                  # s_altkey_data: df_alt,
                  s_altkey_altid: 'STATEFP',  # column id in the existing df
                  s_altkey_newname: 'State'}  # display name

# for aggregating via summing countries:
d_alt_usnation = {s_altkey_type: s_alttype_sum,
                  s_altkey_altid: 'USTotal',  # column id added to df
                  s_altkey_newname: 'United States'}

# combine all into a dictionary with first-level keys being defined names that
#   relate back to the GIS files from which we draw additional information

d_altagg = {s_altagg_country:{'Continent':d_alt_cont,'World':d_alt_world},
            s_altagg_uscounty:{'States':d_alt_usstates, 'Nation':d_alt_usnation}}