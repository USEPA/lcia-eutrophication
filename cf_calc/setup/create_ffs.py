
# Functions to create the cfg.df_fftot, indexed to geotot, flowable, compartment in
# and populate it with FFtots

# Each FFtot is specified by
# 1) a geotot (comprised of geopaths, comprised of geosteps, comprised of ffi/SDMats
# 2) an emission flowable (e.g., kg N emitted)
# 3) an emission context (e.g., freshwater)

import os
import math
import pickle
from datetime import datetime
from collections import OrderedDict

import pandas as pd
import numpy as np

import calc.setup.config as cfg
import calc.setup.config_geotot as cgt

from calc.setup.input import get_widetable_cols

from calc.inout.files import write_dictionary_to_txt
from calc.inout.files import write_df_to_excel, read_or_save_pickle
from calc.inout.files import make_filepathext

from calc.setup.input import get_all_excel_tables
from calc.setup.create_sdmats import create_all_sdmats
from calc.functions.geotot_to_ffi import get_geostep_and_ids, get_ffi_entry, get_sdmat_entry



def create_df_fftot_structure():
    # get the total number of FFtots into a dataframe:
    #   dataframe based on groupings of geotot, flowables, and emission compartment
    #   we add a few columns, to put (for easy access) some of the info 'buried' in the dictionaries

    # We also want those groupings without an entry in the 'factor direct' column,
    #   'Factor Direct' implies we look up the factor from a file;
    #   we do not calculate this as a true 'geotot'

    # TODO: part of the challenge here is that geotot, flow, and compartment
    #   are not shared when we calculate geotot, but if factors already exist,
    #   they can share these col_values.

    mask_factor_dir = cfg.xltables[cfg.t_calc][cfg.s_calc_factordirect].isnull()

    cfg.df_fftot = pd.DataFrame(cfg.xltables[cfg.t_calc][mask_factor_dir].groupby(
        [cfg.s_calc_geotot,
         cfg.s_calc_flowable,
         cfg.s_calc_comp_emit]).size()
                                )

    cfg.df_fftot.rename(columns={0: 'Count'}, inplace=True)
    # add columns to store the dictionaries and the fftot
    #   (note that the fftot is in the dict, too)
    cfg.df_fftot[cgt.s_geoflowcomp_dict] = np.nan
    cfg.df_fftot[cgt.s_fftot_dict] = np.nan
    print(f'grouped index names: {cfg.df_fftot.index.names}')
    print(cfg.df_fftot)
    print(f'total FFtots = {len(cfg.df_fftot)}; total calcs = {cfg.df_fftot["Count"].sum()}')


def populate_dict_geoflowcomp(gtot, flowable, comp):
    """
    Given a geotot, a flowable, and emission compartment (unique identifiers for the
    *geography* of a calculation,
    populate a structured dictionary (contains ids, fftots, etc.)
    :param gtot: string, geotot name
    :param flowable: string, flowable name
    :param comp: string, emission copmartment
    :return: dictionary for the geotot
    """

    print(f'\n-------Function <populate_dict_geoflowcomp> beginning\n'
          f'Gtot= "{gtot}", flowable= "{flowable}", emit comp = "{comp}"')

    # define lists, and a dictionary,
    # for columns we'll use in bulk to create the geopath dfs
    list_geopath_df_cols = [cgt.flowable_in, cgt.flowable_out,
                            cgt.flowable_next, cgt.flowable_conversion,
                            cgt.comp_start, cgt.comp_end,
                            cgt.unit_start, cgt.unit_end]
    list_geopath_df_cols_excel = [cfg.s_ffi_flow_in, cfg.s_ffi_flow_out,
                                  cfg.s_ffi_flow_next, cfg.s_ffi_flow_convert,
                                  cfg.s_ffi_comp_start, cfg.s_ffi_comp_end,
                                  cfg.s_ffi_units_in, cfg.s_ffi_units_out]
    dict_geopath_df_cols_to_excel = dict(zip(list_geopath_df_cols,
                                             list_geopath_df_cols_excel))

    # here's the main dictionary; we add some empty keys that are filled later
    dgfc = {}

    list_geopaths = get_widetable_cols(index_item=gtot,
                                       wide_table=cfg.xltables[cfg.t_geotot])
    dgfc[cgt.list_geopaths] = list_geopaths
    dgfc[cgt.geopaths] = OrderedDict()
    dgfc[cgt.ff_tot] = {}

    gpcount = 0
    for gpath in list_geopaths:
        gpcount += 1
        print(f'\tGeopath = "{gpcount}", number {gpath} of {len(list_geopaths)}')

        # get shorthand for this dictionary:
        dgfc[cgt.geopaths][gpath] = OrderedDict()
        gp_dict = dgfc[cgt.geopaths][gpath]

        list_geosteps = get_widetable_cols(index_item=gpath,
                                           wide_table=cfg.xltables[cfg.t_geopaths])
        gp_dict[cgt.ff_path] = {}
        gp_dict[cgt.ff_path][cgt.matrix] = 'memory limits...'
        gp_dict[cgt.ff_path][cgt.vector] = np.nan
        gp_dict[cgt.list_geosteps] = list_geosteps

        # create dataframe to store rows = steps; columns = variety of in/out info
        gp_dict[cgt.df_steps] = pd.DataFrame(data=None, index=list_geosteps,
                                             columns=list_geopath_df_cols + [
                                                 cgt.matrix_shape, cgt.matrix_sum])

        gp_dict[cgt.geosteps] = OrderedDict()

        gscount = 0
        cumulative_matrix = np.nan  # reset this to zero before entering geostep calc.
        for gstep in list_geosteps:
            gscount += 1
            print(f'\t\tGeostep = "{gstep}", # {gscount} of {len(list_geosteps)} '
                  f'(in geopath {gpath})')

            # create shorthand for geostep dictionary:
            dgfc[cgt.geopaths][gpath][cgt.geosteps][gstep] = OrderedDict()
            gs_dict = dgfc[cgt.geopaths][gpath][cgt.geosteps][gstep]

            # classify based on 1st letters of the gstep
            if cfg.prefix_geo in gstep:
                gs_dict[cgt.step_type] = cgt.step_geoFFi
            else:  # gstep[0:3] = 'SDM'
                gs_dict[cgt.step_type] = cgt.step_SDM

            # region Get geostep (FFi or SDMat) info: matrix and ids
            # get the matrix, and its associated ids_emit (rows) and ids_recv (columns)
            # note that we return arrays, rather than dataframe, because we could be
            # dealing with large, sparse matrices, which can overwhelm dataframes

            # Get the flowable, which we use to get the specify in the FFi table
            # First time through, flowable is specified by the calculation;
            #   following that, the flowable in is the previous geostep's flowable out
            #
            if gscount == 1:
                # we can write into the df for geopath-level df_steps,
                # which we need to figure out flows once we're past gscount = 1
                this_gstep_flow = flowable
                this_gstep_comp = comp
            else:
                # current flow in is the previous flow out
                # we subtract 2: 1 to get to previous iteration,
                # and another 1 because list_geosteps is 0-indexed
                print('current gp_dict:')
                print(gp_dict[cgt.df_steps])

                this_gstep_flow = gp_dict[cgt.df_steps].loc[list_geosteps[gscount - 2],
                                                            cgt.flowable_next]
                this_gstep_comp = gp_dict[cgt.df_steps].loc[list_geosteps[gscount - 2],
                                                            cgt.comp_end]

            # record the flowable and comp in the df_steps
            gp_dict[cgt.df_steps].loc[gstep, cgt.flowable_in] = this_gstep_flow
            gp_dict[cgt.df_steps].loc[gstep, cgt.comp_start] = this_gstep_comp

            print(f'\tGetting matrix and ids for "{gstep}", "{this_gstep_flow}", '
                  f'"{this_gstep_comp}"')

            # check if we have a geoFFI or SDM; this determines how we get data
            if gs_dict[cgt.step_type] == cgt.step_geoFFi:

                gs_matrix, ids_emit, ids_recv = (
                    get_geostep_and_ids(nameID=gstep,
                                        flow=this_gstep_flow,
                                        emit_compartment=this_gstep_comp,
                                        make_diagonal=True,
                                        return_array=True)
                )

                # populate the df_steps (which resides at the path level)
                # note we translate between column names here (where we are writing)
                # and those in excel (where we are looking up)
                for col in list_geopath_df_cols:
                    gp_dict[cgt.df_steps].loc[gstep, col] = (
                        get_ffi_entry(ffi=gstep,
                                      flowable_in=this_gstep_flow,
                                      emit_comp=this_gstep_comp,
                                      column=dict_geopath_df_cols_to_excel[col])
                    )

            else:  # this is an SDM; no unit conversion, etc.

                # check we are NOT starting the geopath with an SDM
                if gscount == 1:
                    raise KeyError(f'Creating a geopath with first geostep as an SDMat.  '
                                   f'Don\'t do it.  Calling info: '
                                   f'Geotot= "{gtot}", geopath= "{gpath}".')

                # Call 'get_geostep_and_ids' without flowable_in or emit compartment.
                # No flow or comp shunts the data collection into the SDM table.
                gs_matrix, ids_emit, ids_recv = (
                    get_geostep_and_ids(nameID=gstep, return_array=True)
                )

                # the df_steps dataframe gets most of the col_values from the previous row,
                # except unit conversion is set to 1, and we fix the compartments.
                prev_gstep = list_geosteps[gscount - 2]

                gp_dict[cgt.df_steps].loc[gstep, :] = \
                    gp_dict[cgt.df_steps].loc[prev_gstep, :]

                #
                # For the SDMs, flow in is flow out, and conversion is 1
                gp_dict[cgt.df_steps].loc[gstep, cgt.flowable_in] = this_gstep_flow
                gp_dict[cgt.df_steps].loc[gstep, cgt.flowable_out] = this_gstep_flow
                gp_dict[cgt.df_steps].loc[gstep, cgt.flowable_conversion] = 1

                gp_dict[cgt.df_steps].loc[gstep, cgt.comp_start] = (
                    get_sdmat_entry(gstep, cfg.s_sdm_comp_start)
                )

                gp_dict[cgt.df_steps].loc[gstep, cgt.comp_end] = (
                    get_sdmat_entry(gstep, cfg.s_sdm_comp_end)
                )
            # endregion

            # done with getting data for FFi or SDMmat

            if isinstance(gs_matrix, np.ndarray):
                if (np.isnan(gs_matrix)).any():
                    # this shouldn't happen... need to check input files
                    raise ValueError(f'Geostep matrix with nans... '
                                     f'Geotot= "{gtot}", geopath= "{gpath}", '
                                     f'geostep= "{gstep}"')
                    # gs_matrix = np.nan_to_num(gs_matrix)
            if isinstance(gs_matrix, pd.DataFrame):
                if gs_matrix.isna().values.any():
                    raise ValueError(f'Geostep matrix with nans... '
                                     f'Geotot= "{gtot}", geopath= "{gpath}", '
                                     f'geostep= "{gstep}"')
                    # gs_matrix.fillna(0)


            # Add to the df steps those items that do not come from excel:
            gp_dict[cgt.df_steps].loc[gstep, cgt.matrix_shape] = gs_matrix.shape
            gp_dict[cgt.df_steps].loc[gstep, cgt.matrix_sum] = gs_matrix.sum().sum()

            print(f'\n-----  Current df_steps, with geotot="{gtot}", '
                  f'geopath="{gpath}" (num {gpcount}/{len(list_geopaths)}), '
                  f'geostep="{gstep}" (num {gscount}/{len(list_geosteps)})')

            print(gp_dict[cgt.df_steps])

            # assign ids
            gs_dict[cgt.ids_emit] = ids_emit
            gs_dict[cgt.ids_recv] = ids_recv

            # region Matrix multiplication
            # Note that pandas dot multiplier checks that indices are aligned,
            #   so this could be interesting to consider in future.  But...
            # We have large matrices (59k x 59k elements for freshwater,
            #   13k x 13k for air, etc.), and these do not do well with dataframes.
            # So we are manually checking ids

            if gscount == 1:
                # we initialize the cumulative matrix with the current matrix
                cumulative_matrix = gs_matrix

                # record the associated geofile (to which we can later append the FFtot)
                dgfc[cgt.emit_geofile] = cfg.xltables[
                    cfg.t_geoffi].loc[gstep, cfg.s_gffi_GeoIn]

                # record the emission ids at the geopath level
                #   (the first geostep sets the ids for the entire path)
                gp_dict[cgt.ids_emit] = ids_emit

                # And if this is ALSO the first geopath,
                #   populate the df matrix with emission indices:
                if gpcount == 1:
                    dgfc[cgt.ff_tot][cgt.df_paths] = pd.DataFrame(
                        data=None, index=ids_emit, columns=list_geopaths
                        )

            else:
                # A note about comparing IDs:
                #   We cannot use set comparison... need same elements, same length.
                #   We are comparing current emission (rows) to the previous
                #   (-1, and then another -1 because gscount is 1-indexed) receiving ids.
                #   The == returns a series of True/False, so we check with all()
                #   This comparison works with mixed int/float data types.

                # Get the previous IDs.  Note we have to go 'up' to the path level
                #   to access something in another geostep
                previous_receive_ids = gp_dict[
                    cgt.geosteps][list_geosteps[gscount - 2]][cgt.ids_recv]

                if cumulative_matrix.shape[1] == gs_matrix.shape[0]:
                    # then matrix dimensions correct (col_names of prev matrix = rows of current)

                    # Check equalit; we convert to numpy to make int/float comparison ok.
                    bln_lengths_okay = (len(previous_receive_ids.to_numpy()) ==
                                        len(ids_emit.to_numpy()))

                    bln_ids_match = all(previous_receive_ids.to_numpy() ==
                                        ids_emit.to_numpy())

                    if bln_lengths_okay and bln_ids_match:
                        # if they are equal, we do matrix multiplication and keep the
                        # cumulative result
                        cumulative_matrix = np.matmul(cumulative_matrix,
                                                      gs_matrix)

                        print(f'geostep = {gstep}, num {gscount}/{len(list_geosteps)}, \n'
                              f'\t gs_matrix sum = {gs_matrix.sum().sum()}, and '
                              f'\t cumulative_matrix sum = {cumulative_matrix.sum().sum()}')

                    else:
                        raise KeyError(f'Creating a geopath with mismatched IDs:'
                                       f'Geotot= "{gtot}", geopath= "{gpath}", '
                                       f'geostep= "{gstep}"')
                else:
                    raise KeyError(f'Creating a geopath with mis-sized matrices'
                                   f'Geotot= "{gtot}", geopath= "{gpath}", '
                                   f'geostep= "{gstep}"')
            # endregion

        # Done looping the geosteps ------------
        # Now can do path level calculations

        print(f'\n ---------------- Current df_steps, with geotot="{gtot}", '
              f'geopath="{gpath}" (num {gpcount}/{len(list_geopaths)}), and geostops done ')
        print(gp_dict[cgt.df_steps])

        # Unit conversion - applied to the final matrix
        # We get the conversion factor by multiplying elements of
        #   the unit conversion column of df_steps
        overall_conv_factor = math.prod(
            gp_dict[cgt.df_steps].loc[:, cgt.flowable_conversion])

        if overall_conv_factor == 1:
            pass  # skip any math
        else:
            cumulative_matrix = cumulative_matrix * overall_conv_factor

        # very slow to sum large matrices along axis = 1
        # vect = mat.sum(axis=1)

        if isinstance(cumulative_matrix, pd.DataFrame):
            vect = pd.DataFrame(data=cumulative_matrix.to_numpy().sum(axis=1),
                                index=gp_dict[cgt.ids_emit],
                                columns=[gpath])
        else:
            vect = pd.DataFrame(data=cumulative_matrix.sum(axis=1),
                                index=gp_dict[cgt.ids_emit],
                                columns=[gpath])

        del cumulative_matrix  # on a bigger, badder computer, we could hang onto this

        # now we have all the geosteps and overall factor, so record
        # gp_dict[cgt.ff_path][cgt.matrix] = cumulative_matrix
        gp_dict[cgt.ff_path][cgt.vector] = vect

        # Add the vector of this path to the geotot-level dataframe of paths
        #   We use this df to add paths together into a total FF
        dgfc[cgt.ff_tot][cgt.df_paths].loc[:, gpath] = vect[gpath]

    # Done looping geopaths -------------

    # now we add columns of the path dataframe to get total FF
    # extract the current df of paths
    df_paths = dgfc[cgt.ff_tot][cgt.df_paths]
    # add a sum column
    df_paths['FF Total'] = df_paths.sum(axis=1)  #TODO add units, or better name to column
    dgfc[cgt.ff_tot][cgt.df_paths] = df_paths
    # save the sum of paths into the vector at the total level
    dgfc[cgt.ff_tot][cgt.vector] = df_paths['FF Total']

    # populate some of the basic info at the geotot level,
    #   Based on first and last geosteps of this path
    #   These col_values will be the same across paths

    gstep = list_geosteps[0]
    dgfc[cgt.flowable_in] = gp_dict[cgt.df_steps].loc[gstep, cgt.flowable_in]
    dgfc[cgt.comp_start] = gp_dict[cgt.df_steps].loc[gstep, cgt.comp_start]
    dgfc[cgt.unit_start] = gp_dict[cgt.df_steps].loc[gstep, cgt.unit_start]
    # we already recorded the emisssion ids at the geopath level
    # dgfc[cgt.ids_emit] = gp_dict[cgt.geosteps][gstep][cgt.ids_emit]

    gstep = list_geosteps[len(list_geosteps) - 1]  # -1 for zero indexing
    dgfc[cgt.flowable_out] = gp_dict[cgt.df_steps].loc[gstep, cgt.flowable_out]
    dgfc[cgt.comp_end] = gp_dict[cgt.df_steps].loc[gstep, cgt.comp_end]
    dgfc[cgt.unit_end] = gp_dict[cgt.df_steps].loc[gstep, cgt.unit_end]
    dgfc[cgt.ids_recv] = gp_dict[cgt.geosteps][gstep][cgt.ids_recv]

    return dgfc
    # end of populate_dict_geoflowcomp()


def create_geoflowcomp_name(geo, flow, comp):
    # eventually, may need to replace the strings here with cfg variables

    s = geo+'_'+flow+'_'+comp
    rep = {'GeoTot': 'FF', 'Flow_': '', 'Comp_': ''}
    for k, v in rep.items():
        s = s.replace(k, v)
    print(f'Clean_geoflowcomp returned "{s}"')
    return s


def calculate_geotot_ffs(df_fftot_store_file='', record_files=False):
    """
    Loop through unique fftots required (based on geotot, flowable, and compartment), and
    update/populate cfg.df_fftot, a global dataframe with fftot information,
    including the fftots themselves
    :param df_fftot_store_file: file name (directory provided here) to find a previously-calculated df_fftot
    :param record_files: boolean; if true, save the calculated df_fftot
    :return: none, but updates cfg.df_fftot
    """
    print(f'Function <calculate_geotot_ffs is running...>')

    fullfileloc = make_filepathext(file=df_fftot_store_file,
                                   path=cfg.dir_geotot,
                                   ext='.pkl')

    if os.path.exists(fullfileloc):
        print(f'Reading existing df_fftot from {fullfileloc}')

        pickleout = read_or_save_pickle(action='read', file=fullfileloc)
        cfg.df_fftot = pickleout[0]

    else:
        # we need to create df_fftot

        if len(cfg.df_fftot) == 0:
            # len is zero when initialized in config
            create_df_fftot_structure()  #

        # now step through rows of df_fftot to
        for idx, _ in cfg.df_fftot.iterrows():
            # index is the multiindex tuples

            # get the current geotot, flowable, and comp_emit
            geotot = idx[list(cfg.df_fftot.index.names).index(cfg.s_calc_geotot)]
            flowable_in = idx[list(cfg.df_fftot.index.names).index(cfg.s_calc_flowable)]
            comp_emit = idx[list(cfg.df_fftot.index.names).index(cfg.s_calc_comp_emit)]

            # get a filename from this combo and see if it exists
            geoflowcomp_name = create_geoflowcomp_name(geotot, flowable_in, comp_emit)

            # get the dictionary for this geotot, flow, comp combination
            cur_dict = populate_dict_geoflowcomp(geotot, flowable_in, comp_emit)

            # record into df_fftot
            cfg.df_fftot.loc[idx, cgt.s_geoflowcomp_dict] = [cur_dict]
            cfg.df_fftot.loc[idx, cgt.s_fftot_dict] = [
                cur_dict[cgt.ff_tot][cgt.vector].to_dict(into=OrderedDict)]

            if record_files:
                # record the individual dictionary, FFtots and FFpaths
                write_dictionary_to_txt(
                    dictionary=cur_dict,
                    fname=make_filepathext(file=geoflowcomp_name + cfg.suffix_time,
                                           path=cfg.dir_geotot,
                                           ext='.txt'))

                write_df_to_excel(df=cur_dict[cgt.ff_tot][cgt.vector],
                                  file=cfg.prefix_geofftot + cfg.suffix_time,
                                  path=cfg.dir_geotot,
                                  ext='.xlsx',
                                  sheet_name=geoflowcomp_name)

                write_df_to_excel(df=cur_dict[cgt.ff_tot][cgt.df_paths],
                                  file=cfg.prefix_geoffpaths + cfg.suffix_time,
                                  path=cfg.dir_geotot,
                                  ext='.xlsx',
                                  sheet_name=geoflowcomp_name)

        # now that we have populated df_fftot, save it
        if record_files:
            read_or_save_pickle(action='save',
                                list_save_vars=[cfg.df_fftot],
                                file=make_filepathext(
                                    file=df_fftot_store_file + cfg.suffix_time,
                                    path=cfg.dir_geotot,
                                    ext='.pkl'))




