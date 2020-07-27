# https://docs.python.org/3/faq/programming.html#how-do-i-share-global-variables-across-modules
# https://stackoverflow.com/questions/13034496/using-global-variables-between-files

from collections import OrderedDict

import pandas as pd


# file locations
# where to store intermediate calculations,
#   e.g., the intersections

suffix_time = ''  # this will be set by datetime upon execution

# paths are relative to package
dir_storage = r'\datasets\Intermed'
dir_output = r'\datasets\Output'

dir_sdms = dir_storage + r'\SDMats'
dir_geotot = dir_storage + r'\Geotot'
prefix_geofftot = 'FFtot_'
prefix_geoffpaths = 'FFpaths_'
dir_intersects = dir_storage + r'\Intersect'
# dir_intersects_shp = dir_intersects + r'\shp'
# dir_intersects_xl = dir_intersects + r'\excel'

# the dictionary of in(put) tables read from excel:
xltables = {}

# dictionary of spatial distribution matrices
# used with t_sdmats
d_sdmats = {}

# dataframe of fftots
df_fftot = pd.DataFrame(data=None)

# OrderedDict of geotots
d_gtot = OrderedDict()


# geofile in/out
gf_join = '_'
gf_prefix_intersect = 'Isect_'


# names for the tables
# note that the string col_values are irrelevant; I've just include some description

t_reg_files = 'table of regular files'
t_geo_files = 'table of geo files'
t_data = 'table of data (specifies source and column or matrix)'
t_sdmats = 'table of spatial dist matrices'
t_calc = 'table of calcs to do'
t_ffi = 'table of individual FFs'  # todo - need better names
t_geoffi = 'table of geoffi'
t_geopaths = 'table of geopaths'
t_geotot = 'table of geotots'
t_proj = 'table of geo projections'
t_stoich = 'table of stoichometric conversions'
t_ics = 'table of impact categories'

# region Dictionary of excel table names to replace
# used in the function to read excel tables (input.py, get_all_excel_tables)
# global replace_table_names
replace_table_names = {'Tbl_RegFiles': t_reg_files,
                       'Tbl_GeoFiles': t_geo_files,
                       'Tbl_Data': t_data,
                       'Tbl_SDM': t_sdmats,
                       'Tbl_FFi': t_ffi,
                       'Tbl_GeoFFi': t_geoffi,
                       'Tbl_GeoPath': t_geopaths,
                       'Tbl_GeoTot': t_geotot,
                       'Tbl_Calc': t_calc,
                       'Tbl_Projections': t_proj,
                       'Tbl_Stoichiometry': t_stoich,
                       'Tbl_ICs':t_ics}
# other names
# 'Tbl_PathsFlows', 'Tbl_ICs', 'Tbl_Flows', 'Tbl_Units',
# 'Tbl_IC_Flow', 'Tbl_IC_GeoTot', 'Tbl_Calc', 'Tbl_Projections'}
# endregion


# region Dictionary of table columns
# First key is table names; this holds a dictionary of my names for the table columns

# For COMMONLY-USED names, set up names here, so a change in excel doesn't force
#  a large renaming.
# However, for infrequently used names, just access them using dictionary keys.


# NOTE: It's important that the t_reg_files and 't_geo_files' share the same names when possible
# the functions that read data from these tables need to be able to refer to the
# base directory in the same way

# the strings here don't have to match anything... and could be jibberish.
# For the share they're just used internally, so I've tried to make them more descriptive

# shared
s_nameID = 'name and ID, first col of each table'
s_id = 'id'
s_namecol = 'name column'
s_basedir = 'base directory'
s_folder = 'subfolder'
s_ext = 'extension'
s_filename = 'filename'


# columns that MATCH Excel files.
xlcols = {t_reg_files: {s_nameID: 'RegFile_NameID',
                        s_basedir: 'Base_Directory',
                        s_folder: 'Folder',
                        s_filename: 'File_Name',
                        s_ext: 'Extension',
                        s_id: 'ID',
                        },
          t_geo_files: {s_nameID: 'GeoFile_NameID',
                        s_basedir: 'Base_Directory',
                        s_folder: 'Folder',
                        s_filename: 'File_Name',
                        s_ext: 'Extension',
                        s_id: 'Header_ID',
                        s_namecol: 'Header_Name'
                        }
          }
          # not done with variables, only really used in one script
#           t_ffi: {s_nameID: 'FFi_NameID',
#                   },
#           t_geopaths: {s_nameID: 'GeoPath_NameID'},
#           t_geotot: {s_nameID: 'GeoTot_NameID'},
#           t_proj: {s_nameID: 'Proj_NameID'},
#           t_calc: {s_units: 'Units'}
#           }
#
# # xlcols['tpath'] = {'name': 'Pathway_NameID'}
# # xlcols[''] = {'s_nameID': '_NameID'}
# # xlcols[''] = {'s_nameID': '_NameID'}
# # xlcols[''] = {'s_nameID': '_NameID'}
# # xlcols[''] = {'s_nameID': '_NameID'}
# # xlcols[''] = {'s_nameID': '_NameID'}
# # xlcols['t_calc'] = {'s_nameID': '_NameID'}
# endregion



# for those not shared, these match excel exactly

# table regular files
s_regfile_xlsheet = 'Excel Sheet'
s_regfile_matname = 'Matlab Name'
s_regfile_ismatrix= 'Is Matrix'
s_regfile_matrixhasIDs= 'Matrix Has IDs'
s_regfile_assocgeo= 'Assoc GeoFile'


# table geofiles
s_geofile_namecol = 'Header_Name'

# table data
s_data_nameID = 'Data_NameID'
s_data_regfilename = 'RegularFile Name'
s_data_colregfile = 'Column_RegFile'
s_data_geofilename = 'GeoFile Name'
s_data_colgeofile = 'Column_Geo'
s_data_proptype = 'Property Type'
s_data_assocgeo = 'Associated GeoFile'



# table of impact categories
s_ics_unitsCFmid = 'Units_CF_mid'
s_ics_ref_target = 'Reference Target'
s_ics_ref_geotot = 'Reference Geotot'
s_ics_ref_flowable = 'Reference Flowable'
s_ics_ref_emitcomp = 'Reference Emission Compartment'
s_ics_ref_sector = 'Reference Sector'


# table sdmats
# Column names written here, so can access directly in code as
# cfg.s_rows_geofile
# Okay as long as not redeclared here
s_sdm_nameID = 'SDM_NameID'
s_sdm_savefile = 'Save File Name'
s_sdm_saveext = 'Save File Extension'
s_sdm_comp_start = 'Start Compartment'
s_sdm_comp_end = 'End Compartment'
s_sdm_use_existing = 'Use_Existing_File'
s_sdm_rowsgeofile = 'Rows_GeoFile'
s_sdm_colsgeofile = 'Cols_GeoFile'
s_sdm_rowsid_files = 'Rows get IDs from data parent file'
s_sdm_colsid_files = 'Cols get IDs from data parent file'
s_sdm_rowsid_data = 'Rows get IDs from Data'
s_sdm_colsid_data = 'Cols get IDs from Data'
s_sdm_isect_val = 'Intersect Value (Data)'
s_sdm_force1to1 = 'Force 1 to 1'
s_sdm_save_sdm = 'Save SDM'


# Table FFi
s_ffi_geoffi = 'GeoFFi'
s_ffi_data = 'FFi Data'
s_ffi_comp_start= 'Start_Compartment'
s_ffi_comp_end= 'End_Compartment'
s_ffi_flow_in= 'Flow In'
s_ffi_flow_out= 'Flow Out'
s_ffi_flow_next = 'Flow Next'
s_ffi_flow_convert = 'Stoichiometric Conversion'
s_ffi_units_in= 'Unit In'
s_ffi_units_out= 'Unit Out'


# table geoffi
s_gffi_GeoIn = 'Geo In'  # was Geo Emit and Receive?
s_gffi_GeoOut = 'Geo Out'

# table data
s_data_propertytype = 'Property Type'
s_datarow_intensive = 'Intensive' # must match the value in the rows (of column s_data_propertytype)


# for calc table:
# column names written here, so can access directly in code as
# cfg.s_calc_ic
s_calc_do = 'Do Calc'
s_calc_id = 'ID Calc'
s_calc_ic = 'Impact Category'
s_calc_flowable = 'Flowable'
s_calc_unit_in = 'Unit Start'
s_calc_geotot= 'GeoTot'
s_calc_factordirect = 'Factor Direct'
s_calc_comp_emit = 'Emit Compartment'
s_calc_comp_recv = 'Receive Compartment'
s_calc_aggreg= 'Aggregation Target'
s_calc_projection= 'Projection'
s_calc_sector= 'Sector'
s_calc_data1= 'Data 1'
s_calc_data2= 'Data 2'
s_calc_data1scale = 'Data 1 Scaling'
s_calc_data2scale = 'Data 2 Scaling'
s_calc_type= 'Calc Type'
s_calc_unit_end = 'Unit End'
s_calc_is_ref = 'Is Reference'
s_calc_normalize = 'Normalize to Reference'
s_calc_unit_norm = 'Unit Norm'
# columns we add
s_calc_geotot_assoc_geofile = 'Geotot assoc. geofile'
s_calc_data1_assoc_geofile = 'Data1 assoc. geofile'
s_calc_data2_assoc_geofile = 'Data2 assoc. geofile'

# columns for use in excel that we drop in script 'update_calc_table'
list_calc_drop_cols = ['New IC','IC Group',
                       'New Agg','Agg Group',
                       'New Flow','Flow Group']

# for unique intersects:
list_calc_unique_intersects = [s_calc_aggreg,
                               s_calc_geotot_assoc_geofile,
                               s_calc_data1_assoc_geofile,
                               s_calc_data2_assoc_geofile,
                               s_calc_projection]

df_unique_intersects = pd.DataFrame(data=None)


# TODO - rename these as above, so we don't have to specify table name when accessing.
# cfg.xltables[cfg.table_name].loc[:, cfg.s_tbl_text]  is cleaner than
# cfg.xltables[cfg.table_name].loc[:, cfg.xlcols[cfg.table_name][cfg.text]]




# columns that we add programmatically:
# we add these columns, but user never sees the names

# geo files
add_geo_id = 'dict IDs'
add_geo_name = 'dict Name'


# dictionary keys for calculation dictionary (in <getsave_intersect>):
dict_calc_filenameID = 'filename key (nameID)'
dict_calc_datanameID = 'dataname key (nameID)'
dict_calc_projnameID = 'projection name key (nameID)'
dict_calc_flownameID = 'flowable name key (nameID)'


# prefixes
# geosteps: when creating dictionaries to store the geotots,
# we need to know if geosteps are FFs, in which case they have prefix 'GeoFFi' or are SDMs
prefix_geo = 'GeoFFi'


# names used for dealing with projections
proj_crs_wkt = 'well-known text'
proj_crs_code = 'authority code'
proj_crs_proj = 'proj'

# proj_s_code_WCEA = 'ESRI:54034'
# proj_conv_WCEA = 1/1e6

# Mollweide
# proj_crs_default = proj_crs_wkt
# proj_s_default = r'PROJCS["World_Mollweide",GEOGCS["GCS_WGS_1984",DATUM["WGS_1984",SPHEROID["WGS_1984",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["Degree",0.017453292519943295]],PROJECTION["Mollweide"],PARAMETER["False_Easting",0],PARAMETER["False_Northing",0],PARAMETER["Central_Meridian",0],UNIT["Meter",1],AUTHORITY["EPSG","54009"]]'
# proj_conv_default = 1 / 1e6  #mollweide is meters, so we multiply by this to get km2

# code not recognized by osgeo
# # lambert / world cylindrical equal area
proj_crs_default = proj_crs_wkt
proj_s_default = r'PROJCS["World_Cylindrical_Equal_Area",GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,298.257223563]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],PROJECTION["Cylindrical_Equal_Area"],PARAMETER["false_easting",0.0],PARAMETER["false_northing",0.0],PARAMETER["central_meridian",0.0],PARAMETER["standard_parallel_1",0.0],UNIT["Meter",1.0]]'
proj_conv_default = 1 / 1e6  # meters, so we multiply by this to get km2




