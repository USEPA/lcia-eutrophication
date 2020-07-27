
# python file to create spatial distribution matrices, SDMs
# SDMs defined in excel file, table Tbl_SDM

# SDM is an intermediate matrix used in pathway calculations
# FF1 x SDM
# SDM is used to translate the columns of FF1 into a different spatial resolution,
#  e.g., from receiving grid cells to receiving LMEs.

# calculation depends on whether properties being 'translated' are intensive or extensive


# "C:\CalcDir\Python\CalcCFs\testing\Test SDMs.html"
# from calc.functions.get_matrices import create_sdm_from_df2


#


import pandas as pd
import numpy as np

import pickle
import os
import calc.setup.config as cfg
from calc.setup.input import get_data, get_data_fromfiles, get_filetablecol
from calc.setup.input import getput_idsnames_fromfiles

from calc.functions.gis import read_shapefile_fromfile, project_shapefile
from calc.functions.gis import calc_area, multiintersect
from calc.functions.sdmat import create_sdm_from_df2

from calc.inout.files import save_df_or_array, read_or_save_pickle
from calc.inout.files import make_filepathext

# functions to check that table entries are not empty,
# and that they don't have empty strings or zeros.
# this is useful only because many of the 'x's are nested table/list calls


def checkstr(x): return x is not None and len(x) > 0


def checknum(x): return x is not None and x > 0


def checkbln(x): return x is not None and x


def create_all_sdmats(sdmat_store_file=''):
    """
    Run through the table of spatial distribution matrices (t_sdmats);
    for each row, create an sdm
    cfg.sdms is an empty dict created in config.py
    :param sdmat_store_file: name of file (just name, no directory) to store sd_mats
    for later retrieval.  If empty, do not read/save.  If providede we're reading or saving.
    :return: no return, but does save file if name provided, and updates cfg.sdmats
    """

    print(f'Function <create_all_sdmats> running...')

    fullfilepath = make_filepathext(
        file=sdmat_store_file,
        path=cfg.dir_sdms,
        ext = '.pkl')


    if sdmat_store_file != '':
        # if sdmat_store_file is a file, load sdmats from there:
        if os.path.exists(fullfilepath):
            print(f'\tFunction <create_all_sdmats> reading '
                  f'd_sdmats from {fullfilepath}...')
            pickleout = read_or_save_pickle(action='read',
                                            file=fullfilepath)
            cfg.d_sdmats = pickleout[0]

            bln_have = True
            bln_save = False
        else:
            bln_have = False
            bln_save = True

    else:
        bln_have = False
        bln_save = False
        # Not saving the full d_sdmats,
        #   but might still save individuals in the create_sdmats function

    if not bln_have:
        # we create them
        print(f'\tFunction <create_all_sdmats> creating sdmats.')
        for idx, row in cfg.xltables[cfg.t_sdmats].iterrows():

            print(f'\t\t...working on SDMat row = {idx}')

            # assign it to dictionary d_sdmats
            cfg.d_sdmats[idx] = create_sdmat(datarow=row, manual_save_override=False)

        if bln_save:
            read_or_save_pickle(action='save',
                                list_save_vars=[cfg.d_sdmats],
                                file=make_filepathext(file=sdmat_store_file,
                                                      path=cfg.dir_sdms,
                                                      ext='.pkl'))

    else:
        print(f'\t\t...<create_all_sdmats> done')
        print(f'\t\tcfg.d_sdmats keys: {cfg.d_sdmats.keys()}')


def create_sdmat_from_name(nameID, manual_save=False):

    if nameID in cfg.xltables[cfg.t_sdmats].index:
        datarow = cfg.xltables[cfg.t_sdmats].loc[nameID]

        sdmat = create_sdmat(datarow=datarow, manual_save_override=manual_save)
        if not nameID in cfg.d_sdmats.keys():
            cfg.d_sdmats[nameID] = sdmat

        return sdmat

    else:
        raise KeyError(f'Function <create_sdmat_from_name> called with '
                       f'name = {nameID} that is not in table of sdmats')



def create_sdmat(datarow, manual_save_override=False):
    """
    For a given row from table t_sdmats, create the sd_mat, depending on type of calc.
    Calculations may be only area-based, may put col_values to 1 or some other value
    (see called function "create_sdm_from_df2")
    :param datarow: a series extracted from table t_sdmats
    :param manual_save_override: if true, prevent from saving
    :return: a dataframe sdmat, based on parameters in the table
    """


    print(f'\t\tFunction <create_sdmat> starting...')

    s_rows = 'rows'
    s_cols = 'col_names'
    s_vals = 'vals'
    
    # If the 'Use_Existing_File' column is not empty and not an empty string,
    #  read the file directly and return it.  (And thus end the function)
    if checkstr(datarow[cfg.s_sdm_use_existing]):
        getfile = datarow[cfg.s_sdm_use_existing]
        print(f'\t\t\tFunction <create_sdmat> is returning '
              f'existing file= {getfile}')
        return get_data_fromfiles(filenameID=getfile,
                                  return_array=False)

    bln_have_geo = checkstr(datarow[cfg.s_sdm_rowsgeofile]) and \
                   checkstr(datarow[cfg.s_sdm_colsgeofile])

    bln_have_data = checkstr(datarow[cfg.s_sdm_rowsid_data]) and \
                    checkstr(datarow[cfg.s_sdm_colsid_data])

    bln_have_files = checkstr(datarow[cfg.s_sdm_rowsid_files]) and \
                     checkstr(datarow[cfg.s_sdm_colsid_files])

    bln_calc_by_value = bln_have_data or bln_have_files

    bln_have_vals = checkstr(datarow[cfg.s_sdm_isect_val])
    bln_1to1 = checkbln(datarow[cfg.s_sdm_force1to1])

    if manual_save_override:
        bln_save = manual_save_override
    else:
        bln_save = datarow[cfg.s_sdm_save_sdm]

    if not bln_have_geo:
        raise TypeError(f'Function <create_sdmat> called without geofiles, '
                        f'which provide the master IDs. Tbl_SDM row = '
                        f'{datarow[cfg.t_sdmats][cfg.s_nameID]}')

    if bln_calc_by_value and not (bln_have_vals or bln_1to1):
        raise TypeError(f'function <create_sdmat>, called with data files'
                        f'specified, but no value column or 1to1.  Tbl_SDM row =  '
                           f'name = {datarow[cfg.t_sdmats][cfg.s_nameID]}')

    # if data are specified, we use the geofiles only for ids;

    # rows and column geofile(s) must be specified, read them
    # we'll definitely need IDs, and we may perform intersect
    master_rowids, _ = getput_idsnames_fromfiles(
        filenameID=datarow[cfg.s_sdm_rowsgeofile],table=cfg.t_geo_files
    )
    master_colids, _ = getput_idsnames_fromfiles(
        filenameID=datarow[cfg.s_sdm_colsgeofile],table=cfg.t_geo_files
    )

    if bln_calc_by_value:


        if bln_have_data:
            # we get ids from a column in a named datafile
            rowids = get_data(datanameID=datarow[cfg.s_sdm_rowsid_data])
            colids = get_data(datanameID=datarow[cfg.s_sdm_colsid_data])

        else:  # bln_have_files:
            # we get ids by loking up the ids associated with a file

            f, t, _ = get_filetablecol(datanameID=datarow[cfg.s_sdm_rowsid_files])
            rowids = getput_idsnames_fromfiles(filenameID=f, table=t)
            f, t, _ = get_filetablecol(datanameID=datarow[cfg.s_sdm_colsid_files])
            colids = getput_idsnames_fromfiles(filenameID=f, table=t)

        # if col_values are specified, read them:
        if bln_have_vals:
            # get the col_values, but return the array in case there are repeated
            # index col_values in the ID column (e.g., for News2_FracFW).
            # The repeated index col_values seem to mess with the order of the col_values,
            # which is critical in this case.
            vals = get_data(datarow[cfg.s_sdm_isect_val], return_array=True)

        else:
            # there is no else; we checked that if we have data ids, we have
            # either vals or it's 1to1, which we set in the sdm function
            vals = np.ones(rowids.shape)  # assumed the same as colids.shape

        # the way the table is set up, rowids, colids, and value must have same length
        df_data = pd.DataFrame.from_dict(data={s_rows: rowids.to_numpy(),
                                          s_cols: colids.to_numpy(),
                                          s_vals: vals})

        sdm = create_sdm_from_df2(dframe=df_data,
                                  col_i=s_rows, col_j=s_cols,
                                  col_vals=s_vals,
                                  force1to1= bln_1to1)

        # reindex the rows and columns using the master ids (below)
        # necessary because the data from which we pulled info may not have had all ids
        # this may 'unsort' the col_values

    else:
        # We're operating on the geofile.
        # We do an intersection to create the df3cols

        list_file_position = [cfg.s_sdm_rowsgeofile, cfg.s_sdm_colsgeofile]
        list_new_id_names = [s_rows, s_cols]
        list_new_area_names = ['area_rows', 'area_cols']  # strings used here only
        shps = []

        for i in range(0, len(list_new_id_names)):
            # get file
            tempshp = read_shapefile_fromfile(
                geofilenameID=datarow[list_file_position[i]],
                new_id_col=list_new_id_names[i]
            )

            # TODO have function for this in input.py 'try_convert...'

            # convert to the ids to integer, if we can
            try:
                tempshp[list_new_id_names[i]] = (
                    tempshp[list_new_id_names[i]].astype(int)
                )
            except ValueError:
                # if we cannot convert to int, we get a value error
                pass

            # project it
            tempshp = project_shapefile(shp=tempshp,
                                        projection_type=cfg.proj_crs_default,
                                        projection_string=cfg.proj_s_default)
            # calc area
            calc_area(shp=tempshp, new_area_name=list_new_area_names[i],
                      conversion_factor=cfg.proj_conv_default)

            shps.append(tempshp)

        # run the intersection of the geo...
        # union to keep all areas
        # TODO: if shapefiles are the same... avoid intersect?
        shp_intersected = multiintersect(list_shapes=shps, how='union',
                                         new_area_col='area_isct',
                                         new_area_conversion=cfg.proj_conv_default)


        # in the context of these d_sdmats, we are translating an extensive property (area)
        # from one geometry to another, so we normalize by the area of the rows (i)
        # In this SDM, the sum of each row is < = 1.
        sdm = create_sdm_from_df2(dframe=pd.DataFrame(
            shp_intersected.drop(columns='geometry')),
            col_i=list_new_id_names[0],
            col_j=list_new_id_names[1],
            col_i_area=list_new_area_names[0],
            col_j_area=list_new_area_names[1],
            col_intersect_area='area_isct',
            divide_area_by='i'
        )

    # reindex both
    sdm = sdm.reindex(index=master_rowids, fill_value=0)
    sdm = sdm.reindex(columns=master_colids, fill_value=0)

    # because the geofile indices may not be sorted...
    # we have to sort, whether we create from geofiles or data/files
    # https://stackoverflow.com/questions/17315881/how-can-i-check-if-a-pandas-dataframes-index-is-sorted
    # TODO - could add a check to see if index is sorted
    sdm.sort_index(axis='index', inplace=True)
    sdm.sort_index(axis='columns', inplace=True)

    if bln_save:
        # because the column names are likely shared, we reference them
        # using the cfg.xlcols dictionary
        save_df_or_array(data=sdm,
                         path=cfg.dir_sdms,
                         filename=datarow[cfg.s_sdm_savefile],
                         extension=datarow[cfg.s_sdm_saveext])



    return sdm
