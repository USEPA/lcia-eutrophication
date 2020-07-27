import pandas as pd
import geopandas as gpd
import numpy as np

import os

import calc.setup.config as cfg
import calc.setup.config_geotot as cgt
import calc.setup.config_aggreg as cagg

from calc.setup.input import get_data
from calc.setup.input import getput_idsnames_fromfiles

from calc.setup.create_intersects import get_assoc_geofile_from_geotot
from calc.setup.create_intersects import getsave_intersect, get_calc_geo_dict
from calc.setup.create_intersects import update_calc_table

from calc.functions.weighted_avg import weightedaverage
from calc.functions.dframes import return_mask_column_conditions, return_mask_index_conditions

from calc.inout.files import read_or_save_pickle


def checkstr(x): return x is not None and len(x) > 0


def normalize(df_output):

    # Operate on copy of entire cagg.df_output: we subset, and for each subset,
    #   divide by appropriate reference substance and put correct units into Units Final

    # Set the 'is norm ref column' to false prior to creating the df_norm
    #   (which will share this, until we overwrite a few places)
    df_output.loc[:, cagg.str_outputcol_is_normref] = False

    # create a copy of df_output
    df_norm = pd.DataFrame(data=df_output.to_numpy(), index=df_output.index,
                           columns=df_output.columns)

    # clear the target value and unit final
    df_norm.loc[:,cagg.str_outputcol_unitfinal] = ''
    df_norm.loc[:,cagg.str_outputcol_is_normalized] = True
    df_norm.loc[:,cagg.str_aggcol_avg_value] = np.nan

    # set df_output normalization = false
    df_output.loc[:, cagg.str_outputcol_is_normalized] = False


    # unique normalizations are defined by
    #   Geotot, flowable, Emission Compartment
    #   [cfg.s_calc_geotot, cfg.s_calc_flowable, cfg.s_calc_comp_emit]

    list_normalized_specify = [cfg.s_calc_ic]

    # list of t_ics column names to specify the reference
    list_ref_specify_ics = [cfg.s_ics_ref_geotot,
                        cfg.s_ics_ref_flowable, cfg.s_ics_ref_emitcomp,
                        cfg.s_ics_ref_target, cfg.s_ics_ref_sector]

    # list of calc column names to specify the reference
    list_ref_specify_calc = [cfg.s_calc_geotot, cfg.s_calc_flowable,
                             cfg.s_calc_comp_emit, cfg.s_calc_aggreg,
                             cfg.s_calc_sector]


    df_unique_norm = pd.DataFrame(df_output.groupby(
        list_normalized_specify
        ).size())
    df_unique_norm.column='Count'

    # set some empty columns:
    df_unique_norm[cfg.s_ics_ref_target] = ''
    df_unique_norm[cfg.s_ics_ref_sector] = ''
    df_unique_norm[cfg.s_ics_unitsCFmid] = ''

    # For each row, find the matching row in t_ics, and
    #   get the additional data we need to normalize: the target and sector
    tbl_ics = cfg.xltables[cfg.t_ics]

    for idx, row in df_unique_norm.iterrows():
        values = idx
        row_ics = tbl_ics.loc[values,:]
        # row_ics = tbl_ics.loc[
        #           return_mask_column_conditions(
        #               df=tbl_ics,
        #               col_names=[cfg.s_ics_ref_geotot, cfg.s_ics_ref_flowable,
        #                     cfg.s_ics_ref_emitcomp],
        #               col_values=col_values),
        #           :]
        # set col_values in df_unique_norm:

        for c in list_ref_specify_ics:
            df_unique_norm.loc[idx, c] = row_ics[c]

        # also record unit:
        df_unique_norm.loc[idx, cagg.str_outputcol_unitfinal] = row_ics[cfg.s_ics_unitsCFmid]

        #
        # df_unique_norm.loc[idx, cfg.s_ics_ref_geotot] = row_ics[cfg.s_ics_ref_geotot]
        # df_unique_norm.loc[idx, cfg.s_ics_ref_flowable] = row_ics[cfg.s_ics_ref_flowable]
        # df_unique_norm.loc[idx, cfg.s_ics_ref_emitcomp] = row_ics[cfg.s_ics_ref_emitcomp]
        #
        # df_unique_norm.loc[idx, cfg.s_ics_ref_target] = row_ics[cfg.s_ics_ref_target]
        # df_unique_norm.loc[idx, cfg.s_ics_ref_sector] = row_ics[cfg.s_ics_ref_sector]
        # df_unique_norm.loc[idx, cfg.s_ics_unitsCFmid] = row_ics[cfg.s_ics_unitsCFmid]

        # Get normalization value
        # We need to find the one row in the df_output that matches our conditions,
        #   which are the indices of df_unique_norm and the columns we just retrieved

        # mask_ic = return_mask_index_and_col_conditions(
        #     df=df_output,
        #     index_names=[cfg.s_calc_geotot, cfg.s_calc_flowable,
        #                  cfg.s_calc_comp_emit],
        #     index_values=list(idx),
        #     col_names=[cfg.s_ics_ref_target, cfg.s_ics_ref_sector,
        #                cfg.s_ics_unitsCFmid],
        #     col_values=row_ics[cfg.s_ics_ref_target, cfg.s_ics_ref_sector,
        #                        cfg.s_ics_unitsCFmid]
        #     )


        mask_ref = return_mask_index_conditions (
            df=df_output,
            index_names=list_ref_specify_calc,
            index_values=row_ics[list_ref_specify_ics].to_numpy().tolist()
            )

        # This mask should return only one True - i.e.,
        #   there should be only one reference FF, flow, compartment per impact category

        # np.size(a) - np.count_nonzero(a)

        if np.count_nonzero(mask_ref) == 0:
            print(f'Function <normalize> found 0 matches for '
                  f'Geotot, Flow, Emit = {idx}')
        elif np.count_nonzero(mask_ref) > 1:
            print(f'Function <normalize> found multiple matches for '
                  f'Geotot, Flow, Emit = {idx}')
        else:
            # just right: one reference substance

            norm_value_this_ic = df_output.loc[mask_ref,
                                               cagg.str_aggcol_avg_value].to_numpy()

            # for this one entry set the 'is normalization ref' to True
            #   (in both norm and output)
            df_norm.loc[mask_ref,cagg.str_outputcol_is_normref] = True
            df_output.loc[mask_ref, cagg.str_outputcol_is_normref] = True

            # ---- now, normalize all others

            # get the locations of rows that share the impact category
            mask_this_ic = return_mask_index_conditions(
                df=df_norm,
                index_names=list_normalized_specify,
                index_values=[idx]
                )

            # write proper units (this is read from the excel table of impact cats)
            df_norm.loc[mask_this_ic,cagg.str_outputcol_unitfinal] = \
                df_unique_norm.loc[idx, cagg.str_outputcol_unitfinal]

            # record reference substance (join together several values)
            df_norm.loc[mask_this_ic, cagg.str_outputcol_normref_descrip] = \
                ', '.join(row_ics[list_ref_specify_ics].to_numpy().tolist())

            # calculate normalized col_values (divide by the normalization value
            #   for the reference flow/compartment/etc.)
            df_norm.loc[mask_this_ic, cagg.str_aggcol_avg_value] = (
                df_output.loc[mask_this_ic, cagg.str_aggcol_avg_value] / norm_value_this_ic
            )

    # done looping impact categories
    # df_norm.to_excel(r'C:\temp\dfnorm.xlsx')
    # df_output.to_excel(r'C:\temp\dfoutput.xlsx')

    # concat normalized onto df_output
    df_output = pd.concat([df_output, df_norm],axis=0)

    return df_output

def add_output_names(df_aggregated, bool_agg_up):

    # Take a df_aggregated, and add col_values into the names column, using either:
    # If aggregation was from a base geofile, use the names in that file, or
    # if we have aggregated up, copy the names used as new targets.

    if bool_agg_up:
        # copy names from ID to name column
        df_aggregated[cagg.str_outputcol_nametarget] = \
            df_aggregated[cagg.shp_list_ids[cagg.idx_target]]

    else:
        # lookup names and merge in
        # The target is a geofile, so get the IDs and names
        # All rows share the same value so take the first row.

        # emission_geotot = df_aggregated.index.get_level_values(cfg.s_calc_geotot)[0]
        # emission_geofile = get_assoc_geofile_from_geotot(emission_geotot)
        target_geofile = df_aggregated.index.get_level_values(cfg.s_calc_aggreg)[0]

        # get the names as a dataframe, indexed by ID
        _, df_names = getput_idsnames_fromfiles(filenameID=target_geofile,
                                                table=cfg.t_geo_files,
                                                return_dict=False)

        # need to set names to match the aggregated:
        # First, the index name of df_names should be the ID column in the aggregated
        df_names.index.rename(cagg.shp_list_ids[cagg.idx_target], inplace=True)
        # Next, the column name of df_names should be the name column
        df_names.columns = [cagg.str_outputcol_nametarget]

        # Update matching columns (the name target) by the matching index (the id_target)
        # Cannot do simple update, because the target ID (index of names)
        # is in the columns of df_aggregated
        # df_aggregated.update(df_names, overwrite=True)

        df_names_reorder = df_names.reindex(
            df_aggregated.loc[:, cagg.shp_list_ids[cagg.idx_target]])
        df_aggregated.loc[:, cagg.str_outputcol_nametarget] = df_names_reorder.to_numpy()

    return df_aggregated



def prep_alternate_aggreg(df_base_intersect, dict_new_agg):
    """
    Adjust the df_intersect to account for aggregating to different levels
    (e.g., from counties up to states)
    :param df_base_intersect: an intersection dataframe, with appropriately named
    target, source, weight ids, names, and areas
    :param dict_new_agg: an aggregation dictionary
    :return: modified df_intersect
    """

    this_type = dict_new_agg[cagg.s_altkey_type]
    this_altid = dict_new_agg[cagg.s_altkey_altid]
    this_newname = dict_new_agg[cagg.s_altkey_newname]

    # get a base dict, which we'll modify
    newagg_weightedavg_dict = create_weightedavg_dict()

    if this_type == cagg.s_alttype_exist:
        # we have existing column that we can aggregate to (e.g., Continents included in global country file)

        # send the df_base_intersect to aggregation function, but specify different target column
        newagg_weightedavg_dict[cagg.idx_target][cagg.key_id] = this_altid
        newagg_weightedavg_dict[cagg.idx_target][cagg.key_name] = this_newname
        newagg_weightedavg_dict[cagg.idx_target][cagg.key_area] = this_newname + '_area'

        alt_areas = df_base_intersect.groupby(this_altid)[
            cagg.shp_list_areas[cagg.idx_target]].sum()

        # rename
        alt_areas.rename(this_newname + '_area', inplace=True)

        # merge the areas back to df_base_intersect
        df_base_intersect = df_base_intersect.merge(right=alt_areas,
                                                    how='inner',
                                                    left_on=this_altid, right_index=True)


    elif this_type == cagg.s_alttype_sum:
        # we will add up the current target to a single value (e.g., global countries added to a world value)

        # get rows where the current target id has a value; these are the rows
        #   that we wish to include in the sum
        mask_cur_target = ~df_base_intersect[cagg.shp_list_ids[cagg.idx_target]].isnull()

        # add a new column  #okay to modify input column
        df_base_intersect.loc[mask_cur_target, this_altid] = this_newname

        # area of the new column will be the sum of the original (base) target areas
        df_base_intersect.loc[mask_cur_target, this_altid + '_area'] = \
            df_base_intersect.loc[mask_cur_target, cagg.shp_list_areas[cagg.idx_target]].sum()

        # send to aggregation function, with specification of new target id and area
        newagg_weightedavg_dict[cagg.idx_target][cagg.key_id] = this_altid
        newagg_weightedavg_dict[cagg.idx_target][cagg.key_name] = this_newname
        newagg_weightedavg_dict[cagg.idx_target][cagg.key_area]: this_altid + '_area'

    else:
        # this_type = s_alttype_merge
        # Take another file and merge in a column (e.g., country classifications (OECD status) merged to countries)
        pass

    # Done looping through list of new aggregations
    return df_base_intersect, newagg_weightedavg_dict


def get_intensive_extensive(datanameID):
    if datanameID in cfg.xltables[cfg.t_data].index:

        prop_type = cfg.xltables[cfg.t_data].loc[datanameID, cfg.s_data_propertytype]

        if prop_type is not None and prop_type != '':
            return prop_type
        else:
            raise KeyError(f'Function <get_intensive_extensive> did not find a '
                           f'property type (intensive/extensive) associated with '
                           f'datanameID = {datanameID}')
    else:
        raise KeyError(f'Function <get_intensive_extensive> found aggregation data '
                       f'associated with "{datanameID}" that was '
                       f'not in table "{cfg.t_data}".')

    # file, table, _ = get_filetablecol(datanameID)
    #
    # if table == cfg.t_data:
    #     return cfg.xltables[cfg.t_data][cfg.s_data_propertytype]
    #
    # else:


def create_combined_weight_dicts(calc_rowseries):
    # defne combo_dicts, with column name, area, type(intensive or extensive),
    #   and scaling factor

    # So dictionaries look like this:
    # {2: {'area': 'area_weight1',
    #      'scaling': 1.0,
    #      'type': 'Extensive',
    #      'val': 'val_weight1'},
    #  3: {'area': 'area_weight2',
    #      'scaling': 1.0,
    #      'type': 'Extensive',
    #      'val': 'val_weight2'}}


    combo_dicts = {cagg.idx_weight1: {}, cagg.idx_weight2: {}}
    # Create weighting dicts:
    # the name and area are names that have been standardized when the
    #   shapefiles were imported and intersected
    for idx in combo_dicts.keys():
        combo_dicts[int(idx)] = {cagg.key_val: cagg.shp_list_values[int(idx)],
                            cagg.key_area: cagg.shp_list_areas[int(idx)]}

    # we get the intensive/extensive property from the data table:
    combo_dicts[cagg.idx_weight1][
        cagg.key_type] = get_intensive_extensive(calc_rowseries[cfg.s_calc_data1])
    combo_dicts[cagg.idx_weight2][
        cagg.key_type] = get_intensive_extensive(calc_rowseries[cfg.s_calc_data2])

    # read scaling directly from table
    combo_dicts[cagg.idx_weight1][
        cagg.key_scale] = calc_rowseries[cfg.s_calc_data1scale]
    combo_dicts[cagg.idx_weight2][
        cagg.key_scale] = calc_rowseries[cfg.s_calc_data2scale]

    return combo_dicts


def combine_weight_cols(df_in, col1_dict, col2_dict):
    # each combo dict has column name, area, type (intensive or extensive),
    #   and scaling factor
    #
    # cagg.key_name: , cagg.key_area: , cagg.key_type:, cagg.key_scale
    #
    # return a new df, also with a name for the column weight to use,
    #   and its new property (intensive, since we've scaled here)

    # okay to modify original, so we use df_in
    # prepare a column to create; set value to 0
    df_in['weight_use'] = 0

    #
    for i, c_dict in enumerate([col1_dict, col2_dict]):
        print(f'loop number {i}, with col dict keys = {c_dict.items()}')
        #
        if c_dict[cagg.key_type] == cagg.str_extensive:
            df_in['weight' + str(i + 1) + '_int/extensive adjusted'] = (
                    df_in[c_dict[cagg.key_val]] / df_in[c_dict[cagg.key_area]])
        else:
            # no area adjustment
            df_in['weight' + str(i + 1) + '_int/extensive adjusted'] = (
                df_in[c_dict[cagg.key_val]])

        # normalize the column
        df_in['weight' + str(i + 1) + '_norm'] = (
                df_in['weight' + str(i + 1) + '_int/extensive adjusted'] /
                df_in['weight' + str(i + 1) + '_int/extensive adjusted'].max()
        )
        # scale the column
        df_in['weight' + str(i + 1) + '_scale'] = (
                df_in['weight' + str(i + 1) + '_norm'] * c_dict[cagg.key_scale])

        df_in['weight_use'] = (
                df_in['weight_use'] + df_in['weight' + str(i + 1) + '_scale'])

    # done looping columns

    # Return the df, with the new weight column, and the correct type for weighting
    #   Since we've already done the int/extensive adjustment pass back intensive,
    #   to prevent more area scaling
    return df_in, 'weight_use', cfg.s_datarow_intensive


def prep_do_weighted_aggregation(df_isect,
                                 calc_rowseries,
                                 area_weight_all_nan=False,
                                 area_weight_all_0orNan=False,
                                 weightedavg_dict='',
                                 bool_alt_agg=False):
    if weightedavg_dict == '':
        # no weighting was supplied, so this is standard... get the 'standard' aggregation dictionary
        weightedavg_dict = create_weightedavg_dict()

    # check for multiple weights, in which case we adjust
    # we have to gather info about which columns to combine (and how)
    if (calc_rowseries[cfg.s_calc_data2] is not None and
            calc_rowseries[cfg.s_calc_data2] != ''):
        # get weighting dicts to send to a combine function
        combo_dicts = create_combined_weight_dicts(calc_rowseries=calc_rowseries)

        # send to the combine function, and get back updated intersect table,
        #   as well as new info about which weight column to use.
        df_isect, use_weight_val, use_weight_type = combine_weight_cols(
            df_in=df_isect,
            col1_dict=combo_dicts[cagg.idx_weight1],
            col2_dict=combo_dicts[cagg.idx_weight2])

        # We set the weight val to the newly created 'use_weight_val'
        #   Note that the weight id is not used, so no need to adjust
        weightedavg_dict[cagg.idx_weight1][cagg.key_val] = use_weight_val

    else:
        # df_isect is unchanged; get the correct type (for weight 1)
        use_weight_type = get_intensive_extensive(calc_rowseries[cfg.s_calc_data1])

    # Send to aggregation function
    #   First, get a boolean to send to the aggregation function
    bool_weight_is_intensive = use_weight_type == cfg.s_datarow_intensive

    df_agg, _ = weightedaverage(
        dframe=df_isect,
        target_id=weightedavg_dict[cagg.idx_target][cagg.key_id],
        source_value=weightedavg_dict[cagg.idx_source][cagg.key_val],
        target_area=weightedavg_dict[cagg.idx_target][cagg.key_area],
        intersect_area=cagg.shp_isect_area,
        weight_value=weightedavg_dict[cagg.idx_weight1][cagg.key_val],
        weight_area=weightedavg_dict[cagg.idx_weight1][cagg.key_area],
        source_intensive=True,
        weight_intensive=bool_weight_is_intensive,
        set_nan_source_to_zero=True,
        area_weight_where_no_weight=area_weight_all_nan,
        area_weight_where_0orNan=area_weight_all_0orNan,
        verbose=False)

    # print(df_agg.head())

    # If there was alternate aggregation, the target id was changed,
    #   but we still need that column to have a standard name, so
    #   put it back to the 'official' target id

    df_agg.rename(
        columns={weightedavg_dict[cagg.idx_target][cagg.key_id]:
                     cagg.shp_list_ids[cagg.idx_target]},
        inplace=True)

    # send results to be added to big result table
    # expand_df(df_output, df_agg, cols_to_add)
    df_agg_with_multiindex = expand_df(row_series=calc_rowseries,
                                       df=df_agg,
                                       cols_keep_in_df=cagg.list_agg_cols_keep)

    # if doing alternate aggregation, replace the target aggregation in bulk
    if bool_alt_agg:
        # base_aggregation = df_agg_with_multiindex.index.get_level_values(cfg.s_calc_aggreg)[0]
        # df_agg_with_multiindex.rename(
        #     index={base_aggregation,
        #            weightedavg_dict[cagg.idx_target][cagg.key_name]}, inplace=True)
        df_agg_with_multiindex.index.set_levels(
            [weightedavg_dict[cagg.idx_target][cagg.key_name]],
            level=cfg.s_calc_aggreg,
            inplace=True)

    df_agg_for_concat = prep_df_for_concat(df_target=cagg.df_output,
                                           df_incoming=df_agg_with_multiindex,
                                           incoming_cols_keep=cagg.list_agg_cols_keep)

    # if not normalizing, we know final units
    df_agg_for_concat.loc[:, cagg.str_outputcol_unitfinal] = \
        df_agg_for_concat.index.get_level_values(cfg.s_calc_unit_end)

    return df_agg_for_concat


def expand_df(row_series, df, cols_keep_in_df):
    # Create multiindex from the entire row
    #   This will have several entries we drop later
    #   The index repeats the index from the row_series for
    #   as many entries are in the df
    new_multiindex = pd.MultiIndex.from_arrays(
        np.tile(row_series, (len(df), 1)).transpose(),
        names=list(row_series.index))

    # create dataframe from multiindex and the col_values associated with columns to keep
    df_expanded = pd.DataFrame(data=df.loc[:, cols_keep_in_df].to_numpy(),
                               index=new_multiindex,
                               columns=cols_keep_in_df)

    return df_expanded


def prep_df_for_concat(df_target, df_incoming, incoming_cols_keep):
    # df_target and df_incoming share multi-index;
    # df_incoming may have extra (or fewer) multi-level indices.

    # cols_keep_in_df are the (non-multiindex) columns in the df_incoming that we keep

    # check all indices of incoming are present in target:
    if not all(np.in1d(df_target.index.names, df_incoming.index.names)):
        raise KeyError(f'Function <record_df> called with an incoming df with '
                       f'index names not present in target.')

    # fix the index of the incoming...
    # first create a list of those that are not in the target
    mask_matching = np.in1d(df_incoming.index.names, df_target.index.names)
    list_drop = [c for (c, i) in zip(df_incoming.index.names,
                                     mask_matching) if not i]
    # drop unneeded index
    df_incoming.index = df_incoming.index.droplevel(level=list_drop)

    # reorder to match target
    df_incoming = df_incoming.reorder_levels(order=list(df_target.index.names))

    # if there is a set of columns to keep, select those
    df_incoming = df_incoming.loc[:, incoming_cols_keep]

    # add missing columns (concat will do this later,
    #   but we need them now to deal with normalization)
    df_incoming = df_incoming.reindex(columns=df_target.columns)

    return df_incoming


def get_intersection_df(calc_rowseries):
    """
    For a calculation row, return the appropriate dataframe of the geo-intersection.
    This dataframe will have been created with appropriate labels for the
    id, value, and area of the target, source, and weight.
    Weight col_values are then added to the dataframe, and we can perform aggregation.
    :param calc_rowseries: a row (as a series) from the calculation table
    :return: a dataframe of the corresponding intersected geofiles (target, source, weight)
    """

    if not cfg.s_calc_geotot_assoc_geofile in cfg.xltables[cfg.t_calc].columns:
        # then the calc table has not been updated with associated geofiles, which we need
        update_calc_table()

    dict_calc = get_calc_geo_dict(calc_rowseries)

    # getsave_intersect will check if the file exists, and read it, if so.
    # We need to this combination of shapefiles & projection for intersect
    df_intersected = getsave_intersect(dict_calc=dict_calc,
                                       try_read_save=True,
                                       skip_read=False,
                                       skip_save=False)

    if isinstance(df_intersected, gpd.GeoDataFrame):
        df_intersected=pd.DataFrame(df_intersected)


    return df_intersected


def get_fate_factor_df(calc_rowseries):
    """
    Based on calculation row, get the appropriate fate factor.
    Fate factors are either
        1) calculated previously, and we get them from cfg.df_fftot,
            which is indexed by geotot, flowable, and emission compartment.
        2) Or, if there is a 'factor direct' we just look up the fate factor
            from an excel file.

    :param calc_rowseries: a row (as a series) from the calculation table
    :return: dataframe indexed by id of the emission geofile.
            We rename the fate factor as the source_value
    """

    geotot = calc_rowseries[cfg.s_calc_geotot]
    flowable = calc_rowseries[cfg.s_calc_flowable]
    emitcomp = calc_rowseries[cfg.s_calc_comp_emit]
    factordirect = calc_rowseries[cfg.s_calc_factordirect]

    if checkstr(factordirect):
        # The factor is provided directly from a file (i.e., we did not calculate
        #   as part of the geotot-geopath-geostep framework.
        # So look it up directly...
        df_fate_factor = pd.DataFrame(get_data(datanameID=factordirect, return_array=False))

    else:
        # The ff is stored in the cfg.df_fftot that we previously calculated.
        #   Note that it is stored as a dict, so we have to turn it back into a dataframe
        #     DataFrame.from_dict expects the dictionary to be in form
        #     {'col_a': [val_1, val_2], 'col_b': [val_3, val_4]}
        #     unless we set orient = index

        #   eventually, may need to get other items from the big dictionary:
        #   cfg.df_fftot.loc[idx, cgt.s_geoflowcomp_dict] = [cur_dict]
        idx = (geotot, flowable, emitcomp)

        df_fate_factor = pd.DataFrame.from_dict(data=
                                                cfg.df_fftot.loc[idx, cgt.s_fftot_dict],
                                                orient='index')
        # Rebuilding from dictionary (at least as I stored it),
        #   the df comes back with column name 0.

    # Note that the ids are the index, and the single column is the FFs
    #   Therefore, rename to the correct value name for the source:
    df_fate_factor.columns= [cagg.shp_list_values[cagg.idx_source]]
        # rename(columns={0: cagg.shp_list_values[cagg.idx_source]},
        #                          inplace=True)

    # no further processing of fate factor required, so return it.
    return df_fate_factor


# def create_combo_dict(name, area, type, scale):
#     # not a very exciting function but useful for keeping track of which dicts
#     d_combo = {cagg.key_name: name,
#                cagg.key_area: area,
#                cagg.key_type: type, cagg.key_scale: scale}
#     return d_combo


def create_weightedavg_dict():
    # A function to get back a dictionary of
    # keys (level 1) = indices for target, source, weight1 and 2;
    # keys (level 2) = cagg.key_id, value, and area
    # for the four columns we send to aggregation:
    #   target, source, weight1 (and maybe weight2)

    d = {cagg.idx_target: {},
         cagg.idx_source: {},
         cagg.idx_weight1: {},
         cagg.idx_weight2: {}}

    for idx in d.keys():
        d[idx][cagg.key_id] = cagg.shp_list_ids[idx]
        d[idx][cagg.key_area] = cagg.shp_list_areas[idx]
        d[idx][cagg.key_name] = cagg.shp_list_names[idx]
        if idx != cagg.idx_target:
            d[idx][cagg.key_val] = cagg.shp_list_values[idx]

    return d


def create_df_output(normalize=False):
    """
    Sets the df_output (a global variable), using other global parameters
    :param normalize: boolean; if normalizing, we need to add additional columns
    :return: set the global variable df_output with appropriate index and  columns.
    """
    # uses some global parameters, and
    # start from the calc table, keep certain columns, which become the index

    cagg.list_output_cols = [cagg.str_outputcol_unitfinal,
                             cagg.str_outputcol_nametarget] + cagg.list_agg_cols_keep

    # if we normalize results, add normalization columns
    if normalize:
        cagg.list_output_cols = [cagg.str_outputcol_is_normalized,
                                 cagg.str_outputcol_normref_descrip,
                                 cagg.str_outputcol_is_normref] + cagg.list_output_cols

    blank_index = pd.MultiIndex.from_arrays(
        arrays=([[]] * len(cagg.list_calc_cols_for_index)),
        names=list(cagg.list_calc_cols_for_index)
        )

    cagg.df_output = pd.DataFrame(data=None,
                                  index=blank_index,
                                  columns=cagg.list_output_cols)

    # no return, as we have modified global cagg.df_output
