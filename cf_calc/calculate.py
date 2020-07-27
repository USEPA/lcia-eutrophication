

import pandas as pd

import calc.setup.config as cfg
import calc.setup.config_aggreg as cagg
from calc.setup.setup import main_setup

from calc.inout.files import make_filepathext
from calc.functions.aggregation import get_intersection_df, get_fate_factor_df
from calc.functions.aggregation import prep_do_weighted_aggregation, prep_alternate_aggreg
from calc.functions.aggregation import normalize
from calc.functions.aggregation import add_output_names
from calc.setup.create_intersects import update_calc_table


# region --------- SETTINGS -----------------
#
# Set some file locations:
# main excel file, with lists of file locations, data, and the calculations to run

file_settings = r'CalcCFs Setup and Calculation v0k.xlsx'
file_output = r'EPA_CalcCFs_output.xlsx'


# File name to read/save sdmats as single file:
    # this file (as .pkl) will be saved in config-set location
file_sdmats = 'sdmats'

# File name to read/save the fftots
    # this file (as .pkl) will be saved in config-set location
file_fftots = 'fftots'

# Fall back to area-weighting in targets with NO likelihood data?
#   When a target (e.g., a country or state) has no likelihood data
#   (e.g., fertilizer application), should we revert to area-weighting?
#   A flag will be returned in the output table

area_weight_with_nan_likelihood = True
area_weight_with_0orNan_likelihood = True

# Perform upward aggregation?
#   I.e., if we have calculated county-level FFs, we can aggregate (weighted) to
#   state and country without recalculating the gis
#   Parameters defined in config_aggregate.py
aggregate_up = True


# Calculate normalized results?
#   here, we normalize all CFs in an impact category to a single flowable & context.
#   e.g., for marine eutrophication, if N to freshwater is the reference substance,
#   FF_normalized NOx to air = (FF NOx to air)/(FF N to freshwater)
normalize_results = True


# endregion



# region # --------- BASIC SETUP -------------
# Run basic setup, to
# create the excel tables, a global variable in cfg.xltables
# create the sdmats dictionary, a global variable in cfg.d_sdmats

main_setup(main_excel_file=file_settings,
           sdmat_file='', manual_sdmat_skip=True,
           fftot_file=file_fftots, save_fftots=True, manual_fftot_skip=False,
           create_intersects=True,
           bln_normalize=normalize_results)

print(f'cfg.xltables keys: {cfg.xltables.keys()}')

# endregion

# now we have
#   excel files populated; stored in cfg.xltables dictionary
#   ffots created and stored in cfg.df_fftot
#       (indexed by geotot, flowable, emission compartment)
#   intersection .pkl files created (stored in cfg.dir_intersects
#       (based on required calculations: target, source, and aggregation geometries)
#       (these are computationally heavy; geopandas overlay is not fast)


# region -------- CALCULATION ------------------


# To look for intersects, we need the geofile names, not the geotot.
#   Therefore, see if this had been done (it will in create_intersects).
#   If not, do it.
if cfg.s_calc_geotot_assoc_geofile not in cfg.xltables[cfg.t_calc].columns:
    update_calc_table()

df_calc = cfg.xltables[cfg.t_calc]
df_calc = df_calc[df_calc[cfg.s_calc_do] == True]  # keep equal sign b/c could have empty
df_calc.reset_index(inplace=True)


calc_count = 0
for calc_idx, calc_row in df_calc.iterrows():
    calc_count += 1
    print(f'\n-----------\nAggregation calculation # {calc_count} of {len(df_calc)}')
    print(f'IC={calc_row[cfg.s_calc_ic]} , Flowable={calc_row[cfg.s_calc_flowable]}, '
          f'Geotot={calc_row[cfg.s_calc_geotot]}, Target={calc_row[cfg.s_calc_aggreg]}, '
          f'Sector={calc_row[cfg.s_calc_sector]}, Data1={calc_row[cfg.s_calc_data1]}, '
          f'Data2={calc_row[cfg.s_calc_data2]}')

    # Get the relevant intersect data:
    #   This is the table in which the target, source geometry (fate factor), and
    #   aggregation data have been intersected.
    #   (These are computationally-heavy files, and may take multiple hours to calculate.
    #   If they have been pre-calculated, code will use those)
    #   As set up, these do not have the fate factor information attached
    #   (so we can compute them separately and join)
    df_intersect = get_intersection_df(calc_rowseries=calc_row)

    # Get the relevant total fate factor (this is based on the
    #   1) geotot(overall geometry),
    #   2) the flowable in, and
    #   3) the context (here, emission_compartment)
    # Fate factor is returned with appropriate column name (source_values)
    df_fatefactor_tot = get_fate_factor_df(calc_rowseries=calc_row)


    # Merge the fate factor onto the intersect table,
    #   where it becomes the source col_values
    df_intersect = df_intersect.merge(right=df_fatefactor_tot,
                                      how='inner',
                                      left_on=cagg.shp_list_ids[cagg.idx_source],
                                      right_index=True)

    # aggregate and record
    # this populates cagg.df_output
    df_agg = prep_do_weighted_aggregation(
        df_isect=df_intersect,
        calc_rowseries=calc_row,
        area_weight_all_nan=area_weight_with_nan_likelihood,
        area_weight_all_0orNan=area_weight_with_0orNan_likelihood
        )

    # add names to IDs (also above)
    df_agg = add_output_names(df_aggregated=df_agg, bool_agg_up=False)

    # concat; outer join leaves existing columns in the target dataframe blank
    #   if there are not corresponding columns in the incoming
    cagg.df_output = pd.concat([cagg.df_output, df_agg],
                               join='outer')

    # nothing to return; the output is updating cagg.df_output

    if aggregate_up:

        # Find current aggregation target.  If it matches one of the aggregation
        #   keys, then we can proceed.
        #   Get the target from the first row of the calc in the
        base_target =calc_row[cfg.s_calc_aggreg]

        for top_key, top_dict in cagg.d_altagg.items():
            if top_key == base_target:
                # dict new agg has all the aggregations that correspond to this Target
                for agg_key, agg_dict in top_dict.items():
                    # send agg dictionary to prep to get back new weighting dictionary

                    alt_df_intersect, alt_dict_weightedavg = prep_alternate_aggreg(
                        df_base_intersect= df_intersect, dict_new_agg = agg_dict
                        )

                    # send to prep_do_record
                    df_altagg = prep_do_weighted_aggregation(
                        df_isect=alt_df_intersect,
                        calc_rowseries=calc_row,
                        area_weight_all_nan=area_weight_with_nan_likelihood,
                        area_weight_all_0orNan=area_weight_with_0orNan_likelihood,
                        weightedavg_dict=alt_dict_weightedavg,
                        bool_alt_agg=True)

                    # add names to IDs (also above)
                    df_altagg = add_output_names(df_aggregated=df_altagg, bool_agg_up=True)

                    # join to main df_output
                    cagg.df_output = pd.concat([cagg.df_output, df_altagg],
                                               join='outer')


if normalize_results:
    # for each impact category, get the reference substance (and comp, etc.)
    # find the rows in cagg.df_output with the correct IC, flow, comp,
    # find the ONE row of that subset with agg target (WORLD)
    # divide them all by the world value
    # adjust the 'final units' accordingly

    cagg.df_output = normalize(df_output=cagg.df_output)

cagg.df_output.rename(columns=
                          {cagg.shp_list_ids[cagg.idx_target]:'Aggregation Target ID'})

print(f'Writing df_output...')
cagg.df_output.to_excel(make_filepathext(file_output,cfg.dir_output),merge_cells=False)


# endregion



