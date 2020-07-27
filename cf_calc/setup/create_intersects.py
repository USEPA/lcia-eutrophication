
import os
from collections import OrderedDict
from datetime import datetime

import pandas as pd

from calc.setup import config_geotot as cgt, config as cfg
import calc.setup.config_aggreg as cagg

from calc.setup.input import get_widetable_cols # for creating the cgt.d_geotot

from calc.inout.files import read_or_save_pickle, write_df_to_excel
from calc.inout.files import make_filepathext

from calc.functions.cleanup import categorize_dataframe
from calc.functions.gis import read_shapefile_fromdata, read_shapefile_fromfile, \
    buffer0_invalid_geom
from calc.functions.gis import get_assoc_geofile_from_data
from calc.functions.gis import get_projection, multiintersect
from calc.functions.dframes import return_mask_column_conditions


# display
pd.set_option('max_columns', 8)
pd.set_option('max_colwidth', 40)
pd.set_option('display.width', 200)


# region create a dictionary for display to understand structure of geotots
def create_geotot_dict():
    """
    Create a dictionary of geotots -> geopaths -> geosteps, based on input excel data
    :return: none, but updates the global geotot dict cgt.d_geotot
    """
    for gtot in cfg.xltables[cfg.t_geotot].index:
        # print(f'-----------------------\ngtot = "{gtot}"')

        # only the geopath needs to be ordered, but ordered is much nicer for output
        cgt.d_geotot[gtot] = OrderedDict()

        list_geopaths = get_widetable_cols(index_item=gtot,
                                           wide_table=cfg.xltables[cfg.t_geotot])
        cgt.d_geotot[gtot][cgt.list_geopaths] = list_geopaths
        cgt.d_geotot[gtot][cgt.geopaths] = OrderedDict()

        for gpath in list_geopaths:
            # print(f'\tgpath = "{gpath}"')

            cgt.d_geotot[gtot][cgt.geopaths][gpath] = OrderedDict()

            list_geosteps = get_widetable_cols(index_item=gpath,
                                               wide_table=cfg.xltables[cfg.t_geopaths])

            cgt.d_geotot[gtot][cgt.geopaths][gpath][cgt.list_geosteps] = list_geosteps
            cgt.d_geotot[gtot][cgt.geopaths][gpath][cgt.geosteps] = OrderedDict()

            for gstep in list_geosteps:
                # print(f'\t\tgeostep # {gscount} = "{gstep}"')

                cgt.d_geotot[gtot][cgt.geopaths][gpath][cgt.geosteps][gstep] = {}
    # write_dictionary_to_txt(cgt.d_geotot, r'C:\temp\d3')
# endregion


def get_assoc_geofile_from_geotot(gtot_nameID):

    # given a geotot, get the name of the associated geofile (this is a geofilenameID)

    # A bit convoluted, based on data structure (minimizing user input in excel)
    # We have to find a path associated with the geotot,
    #   and then get the first geostep from that path.
    # But this information is stored in cgt.d_geotot

    if gtot_nameID is None:
        return None

    elif gtot_nameID == '':
        return ''

    else:

        # check to see if the dictionary of geotots has been created
        if len(cgt.d_geotot.keys()) == 0:
            create_geotot_dict()

        path = list(cgt.d_geotot[gtot_nameID][cgt.geopaths].keys())[0]
        step = list(cgt.d_geotot[gtot_nameID][cgt.geopaths][path][cgt.geosteps].keys())[0]

        # step is the geoffi, which we can look up in the geoffi table
        geoffi_row = cfg.xltables[cfg.t_geoffi].loc[step, :]

        return geoffi_row[cfg.s_gffi_GeoIn]


def get_calc_geo_dict(calc_row):

    # the calc_row is a (row) series from the calculation table
    # create a dictionary of geo calculations to do for a row (calc_index)

    # the calling function has already identified the correct row,
    #   So we do not need to mask for "Do Calc", etc.

    # The geo calculations are based on associated geofiles, not the actual data names
    #   (e.g., one associated shapefile can have many pieces of data.
    #   so we only want to do the intersection one time)

    # mask_i = cfg.xltables[cfg.t_calc].loc[:, cfg.s_calc_id] == calc_index
    # mask_do = cfg.xltables[cfg.t_calc].loc[:, cfg.s_calc_do]

    # get the row, and squeeze to make a series
    # calc_row = cfg.xltables[cfg.t_calc].loc[calc_index, :].squeeze()

    # the flowable
    file_flowable = calc_row[cfg.s_calc_flowable]

    # the geotot
    file_target = calc_row[cfg.s_calc_aggreg]
    file_geotot = calc_row[cfg.s_calc_geotot_assoc_geofile]
    file_data1 = calc_row[cfg.s_calc_data1_assoc_geofile]
    file_data2 = calc_row[cfg.s_calc_data2_assoc_geofile]

    # return projection, too
    proj = calc_row[cfg.s_calc_projection]

    return {cfg.dict_calc_flownameID: file_flowable,
            cfg.dict_calc_filenameID: (file_target, file_geotot, file_data1, file_data2),
            cfg.dict_calc_datanameID: (None, None, calc_row[cfg.s_calc_data1],
                                       calc_row[cfg.s_calc_data2]),
            cfg.dict_calc_projnameID: proj}


def convert_calc_id_dict_to_string(calc_id_dict, keys):  # where filenames are created
    """
    Given a calculation id dictionary (flow, file, data, projection), create a string describing it.
    This function provides 'centralized' naming based on calculation descriptors
    :param calc_id_dict: a dictionary describing the flow, file, data, and projection
    :param keys: calling function must specify the dicitonary keys (could be a subset);
    based on keys, information is included or excluded
    :return: a string describing the calculation, based on the keys
    """


    str_flownameID = ''
    str_filenameID = ''
    str_datanameID = ''
    str_projnameID = ''

    list_str = [str_flownameID, str_filenameID, str_datanameID, str_projnameID]

    if cfg.dict_calc_flownameID in keys:
        str_flownameID = calc_id_dict[cfg.dict_calc_flownameID] + cfg.gf_join

    if cfg.dict_calc_filenameID in keys:
        theshapes = calc_id_dict[cfg.dict_calc_filenameID]

        # Walk through shapes and add to string.
        #   Note that the 4th value, data2, may be None
        for s in theshapes:
            if (s is not None) and (s != ''):
                str_filenameID += s + cfg.gf_join

    if cfg.dict_calc_projnameID in keys:
        str_projnameID = calc_id_dict[cfg.dict_calc_projnameID]
    list_str = [str_flownameID, str_filenameID, str_datanameID, str_projnameID]
    finalstr =   ''.join([s+'_' for s in list_str if not s == ''])
    if finalstr[-1] == cfg.gf_join:
        finalstr = finalstr[0:-1]

    return finalstr


def getsave_intersect(dict_calc, try_read_save=True, skip_save=False, skip_read=False):
    """
    Based on a calculation dicitionary (flow, files, data, and projection),
    determine the needed geospatial intersection;
    if it is not present in the intermediate folder, create it.
    :param dict_calc: calculation dictionary, with keys for file, data, and projection
    :param try_read_save: boolean - if true, try to read and save the file based on dict info
    :param skip_save:
    :param skip_read:
    :return: intersected shapefile (but may also save along the way)
    """

    shp_fileIDs = dict_calc[cfg.dict_calc_filenameID]
    shp_dataIDs = dict_calc[cfg.dict_calc_datanameID]
    projectionID = dict_calc[cfg.dict_calc_projnameID]

    if not try_read_save:
        bln_have = False
        bln_save = False

    else:
        # convert shps to filename:
        filename = cfg.gf_prefix_intersect + \
                   convert_calc_id_dict_to_string(calc_id_dict=dict_calc,
                                                  keys=[cfg.dict_calc_filenameID,
                                                        cfg.dict_calc_projnameID])

        fullfile = make_filepathext(file=filename, path=cfg.dir_intersects, ext='.pkl')

        if os.path.exists(fullfile):
            bln_have = True
            bln_save = False
        else:
            bln_have = False
            bln_save = True

    if skip_save:
        bln_save = False

    if bln_have:
        print(f'\tAlready have intersect "{filename}" at {fullfile}.  Reading...')
        if skip_read:
            print(f'\t\t... skipped reading.')  #for testing
            return None
        else:
            print(f'\t\t... done reading.')
            pickleout = read_or_save_pickle(action='read',
                                            file=fullfile)
            return pickleout[0]

    if not bln_have:
        list_corrected_shapes = []
        for i, s in enumerate(shp_fileIDs):
            print(f'Looping shapefiles for combos; i={i}, shapefile="{s}", associated value="{shp_dataIDs[i]}"')
            # read in, rename id_vals, etc., reproject, and calculate new area
            # Shapes are in order of source, target, weight1, weight2
            #   0: source, a filename, has ids and area - we'll merge the col_values later
            #   1: target, a filename, has ids and area
            #   2(3): weights, a dataname,  have ids, col_values, and area.

            if i < 2:  # i.e., the source and target
                print(f'Getting corrected shape for "{s}", '
                      f'with projID = {projectionID}')
                corrected_shape = read_shapefile_fromfile(
                    geofilenameID=s,
                    new_id_col=cagg.shp_list_ids[i],
                    new_area_col=cagg.shp_list_areas[i],
                    proj_id=projectionID)
                if not all(corrected_shape.is_valid):
                    print(f'Some invalid geometries in "{s}", so trying buffer(0)')
                    corrected_shape= buffer0_invalid_geom(shp=corrected_shape)
                list_corrected_shapes.append(corrected_shape)
            else:
                if (not shp_dataIDs[i] is None) and (shp_dataIDs[i] != ''):
                    print(f'Getting corrected shape for "{shp_dataIDs[i]}", '
                          f'with projID = {projectionID}')
                    corrected_shape = read_shapefile_fromdata(
                        datanameID=shp_dataIDs[i],
                        new_id_col=cagg.shp_list_ids[i],
                        new_val_col=cagg.shp_list_values[i],
                        new_area_col=cagg.shp_list_areas[i],
                        proj_id=projectionID)
                    if not all(corrected_shape.is_valid):
                        print(f'Some invalid geometries in "{s}", so trying buffer(0)')
                        corrected_shape = buffer0_invalid_geom(shp=corrected_shape)
                    list_corrected_shapes.append(corrected_shape)

        # intersect the list
        _, _, area_conv = get_projection(projectionID)
        shp_intersect = multiintersect(list_shapes=list_corrected_shapes,
                                       how='union',
                                       new_area_col=cagg.shp_isect_area,
                                       new_area_conversion=area_conv)

        if bln_save:
            # save the intersection

            fullfile = make_filepathext(file=filename, path=cfg.dir_intersects)
            print(f'\tSaving intersected shape (-geometry and categorized) to "{fullfile} "...')

            # shp_intersect.to_file(filename=make_filepathext(file=filename, path=cfg.dir_intersects, ext=.shp), driver='ESRI Shapefile')

            shp_intersect.drop(columns=['geometry'], inplace=True)

            # Can save some space by converting string columns to categories
            #   In one test, got 25% reduction in space
            shp_intersect = categorize_dataframe(shp_intersect)

            read_or_save_pickle(action='save', list_save_vars=[shp_intersect],
                                file=filename, path=cfg.dir_intersects, ext='.pkl')

            # write_df_to_excel(df=pd.DataFrame(shp_intersect),
            #                   file=filename, path=cfg.dir_intersects, ext='.xlsx')

            print(f'\t\t... done saving.')
        return shp_intersect


def update_calc_table():

    # Clean up the calculation table a bit:
    #   Replace None with ''
    #   find associated geofiles for certain columns

    # drop unneded columns
    cfg.xltables[cfg.t_calc].drop(columns=cfg.list_calc_drop_cols,inplace=True)

    tbl = cfg.xltables[cfg.t_calc]  # shorthand to work on the table...

    mask_do = tbl[cfg.s_calc_do] == True  # use "==True" to account for blanks

    # Find the associated geofiles.
    #   Empty excel entries in excel are passed back as None; groupby leaves None out,
    #   so we replace None col_values with empty string
    # replace all the Nones
    tbl.fillna(value='', inplace=True)
    #         tbl[list_cols[i]] = replace_in_column(col=tbl.loc[:,list_cols[i]],
    #                           mask_for='None', new_val='')

    list_geocols = [cfg.s_calc_geotot, cfg.s_calc_data1, cfg.s_calc_data2]
    list_new_geofile_cols = [cfg.s_calc_geotot_assoc_geofile, cfg.s_calc_data1_assoc_geofile,
                    cfg.s_calc_data2_assoc_geofile]
    list_funcs = [get_assoc_geofile_from_geotot, get_assoc_geofile_from_data,
                  get_assoc_geofile_from_data]

    # list_masks = {}
    for i in range (0,len(list_geocols)):
        tbl[list_new_geofile_cols[i]] = tbl[list_geocols[i]].apply(list_funcs[i])

        #list_masks[i] = tbl[list_new_geofile_cols[i]] != ''

    # masks = mask_do & (tbl[cfg.s_calc_aggreg] != '') & list_masks[0] & list_masks[1]


def create_df_unique_intersects():
    """
    Get dataframe of the unique intersects (sharing target, geotot, aggregation, and projection)
    :return: none (but updates global dataframe)
    """
    tbl = cfg.xltables[cfg.t_calc]

    # for the mask, we allow data2 to be empty, as it is optional.
    masks = ((tbl[cfg.s_calc_do] == True) & (tbl[cfg.s_calc_aggreg] != '') &
             (tbl[cfg.s_calc_geotot] != '') & (tbl[cfg.s_calc_data1] != '') &
             (tbl[cfg.s_calc_projection] != ''))

    # cols_unique_intersects = [cfg.s_calc_aggreg,
    #                           cfg.s_calc_geotot_assoc_geofile,
    #                           cfg.s_calc_data1_assoc_geofile,
    #                           cfg.s_calc_data2_assoc_geofile,
    #                           cfg.s_calc_projection]

    cfg.df_unique_intersects = pd.DataFrame(
        tbl[masks].groupby(cfg.list_calc_unique_intersects).size())

    cfg.df_unique_intersects.rename(columns={0: 'Count'}, inplace=True)
    print(f'Function <create_save_all_intersects> found {len(cfg.df_unique_intersects)} '
          f'unique intersections to perform (or verify we have). '
          f'Table of unique intersects:\n')
    print(cfg.df_unique_intersects)


def create_save_all_intersects():

    # the getsave_intersect function skips creating files that already exist.
    # File are identified by  geotot, target, data(1 and 2), and projection
    # so we could do a groupby and use that.  e.g.,

    update_calc_table()

    tbl = cfg.xltables[cfg.t_calc]

    create_df_unique_intersects()
    # # for the mask, we allow data2 to be empty
    # masks = ((tbl[cfg.s_calc_do] == True) & (tbl[cfg.s_calc_aggreg] != '') &
    #          (tbl[cfg.s_calc_geotot] != '') & (tbl[cfg.s_calc_data1] != '') &
    #          (tbl[cfg.s_calc_projection] != ''))
    #
    # # cols_unique_intersects = [cfg.s_calc_aggreg,
    # #                           cfg.s_calc_geotot_assoc_geofile,
    # #                           cfg.s_calc_data1_assoc_geofile,
    # #                           cfg.s_calc_data2_assoc_geofile,
    # #                           cfg.s_calc_projection]
    #
    # df_unique_intersects = pd.DataFrame(
    #     tbl[masks].groupby(cfg.list_calc_unique_intersects).size())
    # df_unique_intersects.rename(columns={0: 'Count'}, inplace=True)
    # print(f'Function <create_save_all_intersects> found {len(df_unique_intersects)} '
    #       f'unique intersections to perform (or verify we have). '
    #       f'Table of unique intersects:\n')
    # print(df_unique_intersects)

    # loop through the unique intersections, extract the corresponding calc row,
    #   and perform shapefile intersection.
    idx_count = 0
    for idx in cfg.df_unique_intersects.index:
        # idx is a tuple, in same order as cols_unique_intersects
        idx_count = idx_count+1
        print(f'\n---------\nRunning <create_intersect> # {idx_count} of '
              f'{len(cfg.df_unique_intersects)}; combination = {idx}.\n'
              f'Start time = {datetime.now().strftime("%H:%M")}')

        # link this combination of column col_values to rows in the main calc table
        calc_mask = return_mask_column_conditions(df=tbl,
                                                  col_names=cfg.list_calc_unique_intersects,
                                                  col_values=idx)
        # print(calc_mask)
        # print(tbl[calc_mask])

        # Send the row from calc table for first occurrence of the calc mask
        info = get_calc_geo_dict(calc_row=
                                 cfg.xltables[cfg.t_calc].iloc[
                                 list(calc_mask).index(True), :].squeeze())

        # write the intersect to the file save location
        _ = getsave_intersect(dict_calc=info,
                              try_read_save=True,
                              skip_read=True)  # skip reading if already there


