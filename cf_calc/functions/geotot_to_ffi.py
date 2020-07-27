
# functions for Geotot calculation:
# Geotot is comprised of geopaths, which are comprised of geosteps
# geosteps are either FFi (individual fate factors) or SDMs (spatial distribution matrices)


from collections import OrderedDict

import pandas as pd
import numpy as np

import calc.setup.config as cfg
from calc.setup.input import get_data, fix_data_ids
from calc.setup.input import getput_idsnames_fromfiles
from calc.setup.create_sdmats import create_sdmat_from_name


def checknum(x): return x is not None and x > 0


def get_overall_unit_conv(dict_flows):
    # look through some sort of  of flows and get conversion for each step

    overall_conversion_factor = 1

    dict_convs = OrderedDict()

    if len(dict_flows.keys()) == 1:
        # there's only one flow; no conversion necessary
        dict_convs[1]=1
    else:
        for istep in range(1, len(dict_flows.keys())):
            # print(istep)
            dict_convs[istep] = get_conv_factor(
                flow_from=list(dict_flows.values())[istep],
                flow_to=list(dict_flows.values())[istep + 1]
            )
            overall_conversion_factor = (
                overall_conversion_factor * dict_convs[istep + 1])

    return overall_conversion_factor, dict_convs


def get_geoffi_ids(geo_ffi):
    # need to get corresponding in and out geofiles
    geo_in = cfg.xltables[cfg.t_geoffi].loc[geo_ffi,cfg.s_gffi_GeoIn]
    ids_in = getput_idsnames_fromfiles(filenameID=geo_in,
                                       table=cfg.t_geo_files,
                                       return_dict=False)

    geo_out = cfg.xltables[cfg.t_geoffi].loc[geo_ffi,cfg.s_gffi_GeoOut]
    ids_out = getput_idsnames_fromfiles(filenameID=geo_out,
                                       table=cfg.t_geo_files,
                                       return_dict=False)

    return ids_in, ids_out


def get_conv_factor(flow_from, flow_to):

    if (flow_from in cfg.xltables[cfg.t_stoich].index and
            flow_to in cfg.xltables[cfg.t_stoich].columns
    ):

        factor = cfg.xltables[cfg.t_stoich].loc[flow_from, flow_to]
        if checknum(factor):
            return factor
        else:
            raise ValueError(f'Function <get_conv_factor> got an empty factor '
                             f'for flows '
                             f'({flow_from},{flow_to}) in row or column.')
    else:
        raise KeyError(f'Function <get_conv_factor> called with flows '
                       f'({flow_from},{flow_to})that are not in row or column.')


def get_ffi_geo_ids(ffi, flowable_in, emit_comp):
    geo_ffi = get_ffi_entry(ffi=ffi, flowable_in=flowable_in,
                             emit_comp=emit_comp, column=cfg.s_ffi_geoffi)

    return get_geoffi_ids(geo_ffi=geo_ffi)


def get_ffi_entry(ffi, flowable_in, emit_comp, column):
    ffirow = get_ffi_series(ffi=ffi, flowable_in=flowable_in,
                            emit_comp=emit_comp)
    return ffirow[column]

def get_sdmat_entry(sdmat_id, column):
    sdmat_row = cfg.xltables[cfg.t_sdmats].loc[sdmat_id,:]
    return sdmat_row[column]


def get_ffi_series(ffi, flowable_in, emit_comp):
    # given geoffi and flow in, return the row (as a series)
    mask_geo = cfg.xltables[cfg.t_ffi][cfg.s_ffi_geoffi] == ffi
    mask_flowable = cfg.xltables[cfg.t_ffi][cfg.s_ffi_flow_in] == flowable_in
    mask_emit = cfg.xltables[cfg.t_ffi][cfg.s_ffi_comp_start] == emit_comp

    # should be only one combination
    df_row = cfg.xltables[cfg.t_ffi][mask_geo & mask_flowable & mask_emit]

    if len(df_row) == 0 or len(df_row) > 1:
        raise KeyError(f'Function <get_ffi_series> should have 1 match, '
                       f'but there were {len(df_row)}.  Input variables are  '
                       f'geo_ffi="{ffi}", flow_in="{flowable_in}", and '
                       f'emit compartment="{emit_comp}"')

    return df_row.squeeze()


def get_ids_from_geoffi(geoffi_id):
    # return emit and receive ids based on a geoffi

    geoffi_row = cfg.xltables[cfg.t_geoffi].loc[geoffi_id,:]

    emission_geo = geoffi_row[cfg.s_gffi_GeoIn]
    receive_geo = geoffi_row[cfg.s_gffi_GeoOut]

    emission_ids, _ = getput_idsnames_fromfiles(filenameID=emission_geo,
                                                table = cfg.t_geo_files)
    receive_ids, _ = getput_idsnames_fromfiles(filenameID=receive_geo,
                                                table=cfg.t_geo_files)
    return emission_ids, receive_ids


def get_ffi_data(ffi_id, flow_in, emit, make_diagonal=True, return_array=True):
    # special version of get data...
    # we get data and then (if returning a dataframe) fix based on IDs,
    # and we can also turn to a diagonal matrix

    print(f'\tFunction <get_ffi_data> called with return_array={return_array}')

    ffi_row = get_ffi_series(ffi=ffi_id, flowable_in=flow_in, emit_comp=emit)

    # we read in data *without* ids (i.e., return_array = True)
    datablock = get_data(datanameID=ffi_row[cfg.s_ffi_data], return_array=return_array)

    # get back the dataframe with index matching the parent geofile
    if return_array:
        # we assume that the ids don't need to be fixed, so get from
        # the associated in,out geofiles
        emit_ids, recv_ids = get_ids_from_geoffi(geoffi_id=ffi_row[cfg.s_ffi_geoffi])

        # but the geofiles provide us with a total list; they could have ids in any order,
        # so we can sort these
        emit_ids.sort_values(inplace=True)
        recv_ids.sort_values(inplace=True)

        # though the index of the ids doesn't matter, but for inspection later,
        # it's easier sorted
        emit_ids.reset_index(inplace=True, drop=True)
        recv_ids.reset_index(inplace=True, drop=True)

    else:
        # get associated geofile to check IDs
        data_row = cfg.xltables[cfg.t_data].loc[ffi_row[cfg.s_ffi_data], :]
        assoc_geofile = data_row[cfg.s_data_assocgeo]

        # fix data ids returns a dataframe...
        datablock = fix_data_ids(df=datablock,
                                 ref_geofile=assoc_geofile, sort=True)
        emit_ids = datablock.index
        recv_ids = datablock.columns

    if check_for_onedim(datablock=datablock):
        # this is a 1-D series, etc., so we may diagonalize
        if make_diagonal:
            if return_array:
                # we got an array directly from the file
                datablock = np.diag(datablock)
            else:
                # we got a dataframe back, so we need to diagonalize
                datablock = pd.DataFrame(data=np.diag(datablock.to_numpy()),
                                         index=datablock.index,
                                         columns=datablock.index)
            return datablock, emit_ids, recv_ids
        else:
            # this is 1D and we are not changing
            return datablock, emit_ids

    else:
        # we have a 2D block, and we are returning it
        # no changes needed, as we read in according to return_array
        return datablock, emit_ids, recv_ids



def check_for_onedim(datablock):
    if isinstance(datablock, pd.Series):
        bln_is_oned = True
    elif isinstance(datablock, pd.DataFrame) and datablock.shape[1] == 1:
        bln_is_oned = True
    elif isinstance(datablock, np.ndarray) and datablock.ndim == 1:
        bln_is_oned = True
    else:
        bln_is_oned = False

    return bln_is_oned


def get_geostep_and_ids(nameID, flow='', emit_compartment='',
                        make_diagonal=True, return_array=True):
    # Given an index value from either the SDM or FFi tables,
    #    get back the corresponding matrix and ids

    print(f'\tFunction <get_geostep_and_ids> called with return_array={return_array}')

    # if flow is empty, we have an sdm
    # functions to create sdms check the SDM against the geofile, so the indices are
    # full (no missing col_values) and sorted.
    # note that the SDM is already a matrix, so no need to diagonalize
    # also, SDMs are dataframes, so we can pull off IDs

    if flow == '' and emit_compartment == '':
        if nameID in cfg.d_sdmats:
            # already in the dictionary
            print(f'\t\tgetting sdmat "{nameID}" from sdmat dictionary')
            gs = cfg.d_sdmats[nameID]
        else:
            # not in dictionary; we need to create it
            print(f'\t\tcreating sdmat "{nameID}" and saving to dictionary')
            gs = create_sdmat_from_name(nameID=nameID, manual_save=False)
            print(f'\t\t...done creating {nameID}')

        ids_rows = gs.index
        ids_cols = gs.columns

    else:
        # we are in the table of ffis:
        gs, ids_rows, ids_cols = get_ffi_data(ffi_id=nameID, flow_in=flow,
                                              emit=emit_compartment,
                                              make_diagonal=make_diagonal,
                                              return_array=return_array)

    if return_array:
        if isinstance(gs, pd.DataFrame):
            gs = gs.to_numpy()

    return gs, ids_rows, ids_cols



