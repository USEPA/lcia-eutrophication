import pandas as pd
import numpy as np
import calc.setup.config_aggreg as cagg

def recalculatecolumn(dframe, column_of_ids, old_col, recalc_col):

    """
    Given a data frame with a named ID column (basedOn),
    sum col_values in a recalc_col and put those col_values back into where the old_col was
    """
    temp_recalc = dframe.groupby(column_of_ids).sum()[recalc_col]
    # print (temp_recalc.head())
    temp_recalc.rename(old_col, inplace=True)  # we've recalculated the column; so rename the series to be the old name
    # print (temp_recalc.head())
    # print (temp_recalc.index.dtype)
    dframe.rename(columns={old_col: old_col + '_orig'}, inplace=True)  # in the original data frame, put the old column as X_old
    # print (dframe.columns)
    # print (dframe.info())
    # print (dframe[column_of_ids].dtype)
    dframe = dframe.merge(temp_recalc, how='left', on=column_of_ids) # add the series (i.e., the recalculated 'old' value) back in
    # print (dframe.columns)
    return dframe



def weightedaverage(dframe, target_id, source_value,
                    target_area='', source_area='', intersect_area='',
                    weight_value='', weight_area='',
                    source_intensive=True, weight_intensive=True,
                    area_weight_where_no_weight=False,
                    area_weight_where_0orNan=False,
                    drop_areas_with_no_source=False, weight_id='',
                    set_nan_source_to_zero=False, set_nan_weight_to_zero=False,
                    verbose=False):

    """
    Return a (weighted) average of the SourceValue, based on the TargetID.
    Without weighting, the average is area-based. With a weighting polygon, the average is weighted.  
    
    Combinations:
    - regular average: target_id and source_value
        - example: average a list of col_values (e.g., targetID=1 has col_values 2, 3, 4; targetID2 has col_values 5, 10, 15.
        Then targetID=1 has average 3, and targetID =2 has 10) 
        - we set weights equivalent and assume intensive
        - note that the useAreaIfNoSource may be important here - do you count areas where the source does not reach?
    
    - weighted average: target_id, source_value, and weight_value.  (this is a 'subset' of the spatially-weighted, below)
        - example: average class grade using weights (homework, tests, etc. have different weights; each student is a different targetID)
        
    - spatially-weighted average: target_id, source_value, and weight_value, source_area (if extensive), weight_area (if extensive)
        - example: LCIA calculation, in which countries (targetIDs), a field of FF col_values (sourceValue),
        and a field of likelihoods (weightValues) are intersected.

    - area-weighted average: target_id, source_value, target_area, intersect_area, (source_area if extensive)
        - example: LCIA calculation without weights, in which we assign target col_values based on
        fractions of area occupied by sourceValues
    
    Parameters:
    - df is a dataframe with results of a spatial intersect of at least a target polygon and a source polygon.
        Source and weight must have col_values and areas of the original polygons.
    - targetID,sourceValue, targetArea, sourceArea, weightValue, weightArea, weightID, intersect_area are column names
    - source_intensive and weightIntensive are booleans indicating whether the col_values of source and weight are
        intensive (we do not normalize by area of the source/weight) or extensive (we do normalize)
    - drop_areas_with_no_source: boolean.  What to do when source does not cover part of the target?
        I.e., is a country fate factor affected by areas with no FF? 
        If True: we remove rows with missing source.... *and* we recalculate target & likelihood areas.  
        (Consider a target zone with mostly null FFs, but some portion with an FF.  We wish to use that portion)
        It is for this parameter that weightID might be needed.
    - set_nan_source_to_zero: boolean.  If true, set missing col_values to zero.
    - returnAllTargets: boolean.  If we remove some source col_values, we might also remove some targets... so do we return them all?
    - area_weighting_if_no_weight: boolean.  Used with weight data. If true, for any target polygon with NO weight data, set all weight=1
    - verbose: do we print out as we go along
    
    TO-DO:
    - see in code below
    - make sure difference between NA and 0 col_values is treated correctly.
    - need to make sure all df numbers are float64, in case of division by zero. See https://stackoverflow.com/questions/16244180/handling-zeros-in-pandas-dataframes-column-divisions-in-python
    - if we drop_areas_with_no_source, we recalculate the target areas.  However, we do not recalculate the weight areas... would have to pass an ID to these.
        This will only be a problem if we have weight data that extends beyond source areas (could happen with population and freshwater FF), and if that

    """
    # Set up some booleans to say whether parameters were provided
    # This is not elegant programming, but can make code more readable.
    bln_have_target_area = (target_area != '')
    bln_have_source_area = (source_area != '')
    bln_have_intersect_area = (intersect_area != '')
    bln_have_weight_value = (weight_value != '')
    bln_have_weight_area = (weight_area != '')
    bln_have_weight_id = (weight_id != '')

    if verbose:
        print(f'Optional arguments present?: Target Area = {bln_have_target_area}, '
              f'Source Area = {bln_have_source_area}, '
              f'IntersectArea = {bln_have_intersect_area}, '
              f'WeightValue = {bln_have_weight_value}, '
              f'WeightArea = {bln_have_weight_area}')

    # --- a few checks
    # in cases where we cannot do the calculation, return strings instead of df
    if (not source_intensive) and (not bln_have_source_area or not bln_have_intersect_area):
        # cannot proceed - need source area to make source extensive
        raise KeyError(f'Function <weightedaverage> needs source area '
                       f'to make source extensive')

    if (not weight_intensive) and (not bln_have_weight_area or not bln_have_intersect_area):
        # cannot proceed - need weight area to make weight extensive
        raise KeyError(f'Function <weightedaverage> needs weight area '
                       f'to make weight extensive')

    if area_weight_where_no_weight and ((not intersect_area) or (not target_area)):
        raise KeyError(f'Function <weightedaverage> need intersect and target areas '
                         f'if we plan to do area_weight_where_no_weight')

    # --- set up working data frame
    # create a copy of the data frame to work with.  otherwise, changes here affect the data frame that was passed.
    df = dframe.copy()  # to be explored... memory is saved by not copying, but we wish to leave input dframe alone.

    if verbose:
        print('\n df columns:')
        print(df.columns)

    # --- Adjust drop_areas_with_no_source
    # Have to do this first, as it alters some of the areas.
    if verbose:
        print('\nWorking on dropping areas with NaN source or '
              'setting NaN sources or weights to zero')
        print(f'count of nas in source_value column = {df[source_value].isna().sum()}; '
              f'total rows = {len(df)}')
        # print(df[df[source_value].isna()]

    if drop_areas_with_no_source:
        if verbose:
            print('drop_areas_with_no_source = True; '
                  'removing areas that do not have sourceValues')
        # remove the rows...
        df.dropna(axis=0, subset=[source_value], inplace=True)

        # In case calculation is source extensive or we're doing area-based fractions,
        # we recalculate areas (since we have dropped some data)
        if bln_have_target_area:
            print(df.columns)
            df = recalculatecolumn(dframe=df, column_of_ids=target_id,
                                   old_col=target_area, recalc_col=intersect_area)
            print('modified columns:')
            print(df.columns)
            # print(df[df['NAME']=='Chile'][[target_area,intersect_area]])
        if bln_have_weight_area:
            if not bln_have_weight_id:
                # cannot proceed
                return {'dfAvg': 'need weight ID to recalculate weight areas '
                                 '(b/c drop_areas_with_no_source was called)'}
            else:
                df = recalculatecolumn(dframe=df, column_of_ids=weight_id,
                                       old_col=weight_area, recalc_col=intersect_area)
        if verbose:
            print('df columns after dropping na sources and recalculating new areas')
            print(df.columns)

    if set_nan_source_to_zero:
        # fill source NA col_values with zero (implies than if there is no source, the true value is 0)
        if verbose:
            print('\tset_nan_source_to_zero = True; '
                  'counting areas with no source as source_value = 0 by replacing nans')
        df[source_value] = df[source_value].fillna(0)

    if set_nan_weight_to_zero:
        # fill weight NA col_values with zero (implies true value is 0)
        if verbose:
            print('\tset_nan_weight_to_zero = True; '
                  'counting areas with no weight as weight_value = 0 by replacing nans')
        df[weight_value] = df[weight_value].fillna(0)

    if verbose:
        print(f'\trevised df.... count of nas in sourceValueColumn = '
              f'{df[source_value].isna().sum()}, total rows = {len(df)}')

    # --- Figure out what kind of calculation we're doing, and adjust weight column accordingly
    if not bln_have_weight_value:
        # So we can do calculations in a single framework...
        # since the weight_value name was not supplied, we put in a value that we can access later
        weight_value = 'FunctionSuppliedWeight'
        if (not bln_have_target_area) and (not bln_have_intersect_area):
            # we put equal weights if none supplied
            df[weight_value] = 1
            if verbose:
                print('no weight value, nor area-weighting, supplied, '
                      'so all weights set to 1')
        elif (bln_have_target_area and bln_have_intersect_area):
            # we are doing area-weighting
            df[weight_value] = df[intersect_area] / df[target_area]
            if verbose:
                print('no weight value, but area-weighting info supplied, '
                      'so weights set to area fractions')
        else:
            # cannot proceed
            raise KeyError(f'Function <weightedaverage> need a weight value, '
                           f'target area and intersect areas or none of the above')

    # --- Adjust area_weight_where_no_weight.  Set this parameter and adjuste if needed.
    blnAddedAreaWeights = False

    # prepopulate this, so output is standardized
    df[cagg.str_aggcol_flag_area_weight] = False

    if bln_have_weight_value:
        if area_weight_where_no_weight:
            # We want to do area-weighting in those areas missing weight col_values
            # Generate a series by looking at groups in the target ID where ALL weight col_values are NaN.

            if not area_weight_where_0orNan:
                seriesNoWeights = df.groupby(target_id)[weight_value].apply(
                    lambda x: x.isna().all())
            else:
                # in some circumstances, 0 may indicate data vas not collected
                # (eg., NCEA inventory for P emissions)
                seriesNoWeights = df.groupby(target_id)[weight_value].apply(
                    lambda x: all([v == 0 or np.isnan(v) for v in x]))

            if len(seriesNoWeights) > 0:  # len here counts the trues
                blnAddedAreaWeights = True
                if verbose:
                    print(f'area_weight_where_no_weight was called.  '
                          f'These targetIDs do not have weight col_values:\n'
                          f'{list(seriesNoWeights[seriesNoWeights == True].index)}')
                # reassign weight col_values in the df based on the target IDs with area-weighted col_values.
                # first, get an index
                idx = df[target_id].isin(
                    list(seriesNoWeights[seriesNoWeights == True].index))

                # calculate area-weighted weight col_values
                df.loc[idx, weight_value] = df.loc[idx, intersect_area] / df.loc[
                    idx, target_area]

                # if weight is extensive, we also need to adjust areas for the weights we're adding
                if ~weight_intensive:
                    # set weight area to intersect
                    # when we divide later, this fraction (intersect/weight) will be one.
                    df.loc[idx, weight_area] = df.loc[idx, intersect_area]

                # fix flag column where needed

                df.loc[idx, cagg.str_aggcol_flag_area_weight] = True
                if verbose:
                    print('\t new df, with {} added:'.format(cagg.str_aggcol_flag_area_weight))
                    print(df.loc[idx, :])

        else:
            pass
            # no action; we leave weights as is.

    # -- Check in:
    if verbose:
        print('\nDone adjusting df; here are current columns:')
        print(df.columns)

    # useSourceValue and useWeightValue depend on whether intensive or extensive
    # define some internally-used columns
    useSourceValue = 'useSourceValue'
    useWeightValue = 'useWeightValue'

    if source_intensive:
        df[useSourceValue] = df[source_value]
    else:
        df[useSourceValue] = df[source_value] * (
                df[intersect_area] / df[source_area])
        # if source area = 0?  should return inf or nan...

    if weight_intensive:
        df[useWeightValue] = df[weight_value]
    else:
        df[useWeightValue] = df[weight_value] * (
                df[intersect_area] / df[weight_area])
        # if weight area = 0?  should retun inf or nan...

    # calculate the source x weight
    useSourceXWeight = 'SourceXWeight'
    df[useSourceXWeight] = df[useSourceValue] * df[useWeightValue]
    if verbose:
        print('\ndf after calculating source x weight')
        print(df.head(15))

    # get our sums of the SourcexWeight and WeightValue columns, grouped by target_id
    sums = df.groupby(target_id)[[useSourceXWeight, useWeightValue]].sum()
    if verbose:
        print('\nsums head:')
        print(sums.head(15))

    # set up final data frame
    if verbose: print('\nCreating dfAvg')

    # start from df to make sure we get all targetIDs
    #   Note: cannot sort if there are mixed strings/numbers (where NaN is a number)
    dfAvg = pd.DataFrame(sorted(dframe[target_id].dropna().unique()), columns=[target_id])
    # didn't work for sorting: dfAvg = pd.DataFrame(df[target_id].unique(), columns=[target_id]).sort_values(by = target_id, inplace=True)

    if verbose:
        print('\nBeginning dfAvg (i.e., just IDs:')
        print(dfAvg.head(15))

    # bring the sums column into the new data frame
    dfAvg = dfAvg.merge(sums, how='left', on=target_id)

    # add the flag warning, if it was created
    # if blnAddedAreaWeights:
    # merge using a subset of df, including only the target (as an merge key) and the flag column
    # tempdf = df[[target_id, strFlagAreaWeight]].drop_duplicates(subset = [target_id])
    dfAvg = dfAvg.merge(
        df[[target_id,
            cagg.str_aggcol_flag_area_weight]].drop_duplicates(subset=[target_id]),
        how='left', on=target_id)

    # finally, calculate the average value for the target by dividing the sum(sourceXweight )/ sum(weight)
    dfAvg[cagg.str_aggcol_avg_value] = dfAvg[useSourceXWeight] / dfAvg[useWeightValue]
    if verbose:
        print('\ndfAvg before removing NaNs (if requested):')
        print(dfAvg.head(15))

    return dfAvg, df



# region Test weighted average
#
# dframe = pd.read_csv('W:/work/projects/eutroCFs_EPA_ERG/Mapping/DemoAggregation/TargSourceLikelihood.csv')
# c = ['Argentina', 'Belize', 'Bolivia', 'Canada', 'Chile', 'United States']
# # dfA[dfA['NAME']=='Argentina']
#
# # compare to W:\work\projects\eutroCFs_EPA_ERG\Mapping\DemoAggregation\Spatial Aggregation Demo.xlsx
# # But note that this excel file doesn't have the cases in which missing weights are supplied
#
# # ------------------------------------------------------------------
# # Case 1 - straight average, dummy data
# datacase1 = {'Name': ['A', 'A', 'A', 'B', 'B', 'B'],
#              'Values': [2, 3, 4, 5, 10, 15]}
# dfcase1 = pd.DataFrame(datacase1)
#
# dict_out = weightedaverage(dframe=dfcase1, target_id='Name', source_value='Values')
# dict_out['dfAvg']
#
# #   Name  SourceXWeight  useWeightValue  AvgTargetValue
# # 0    A              9               3             3.0
# # 1    B             30               3            10.0
#
#
#
# # ------------------------------------------------------------------
# # Case 2 - weighted average, dummy data
# datacase2 = {'Name':['A','A','A','B','B','B'],
#             'Values':[2,3,4,5,10,15],
#             'Weights':[1,0,0,0,100,100]}
# dfcase2 = pd.DataFrame(datacase2)
#
# dict_out = weightedaverage(dframe=dfcase2, target_id='Name', source_value='Values',
#                            weight_value='Weights')
# dict_out['dfAvg']
#
# #   Name  SourceXWeight  useWeightValue  AvgTargetValue
# # 0    A              2               1             2.0
# # 1    B           2500             200            12.5
#
#
# # ------------------------------------------------------------------
# # # Case 1 - straight average, real data
# dict_out = weightedaverage(dframe=dframe,
#                            target_id='NAME', source_value='FF_Value',
#                            drop_areas_with_no_source=True)
#
# dfA = dict_out['dfAvg']; dfF = dict_out['dfFull']
# dfA[dfA['NAME'].isin(c)]
#
#
# # ------------------------------------------------------
# # Case 1b - area-weighted average --------------------------------
# # by providing areas of target and intersect, we do area-weighting
#
# dict_out = weightedaverage(dframe=dframe,
#                            target_id='NAME', source_value='FF_Value',
#                            target_area='akm2_targ', intersect_area='akm2_isct',
#                            drop_areas_with_no_source=False, verbose=False)
#
# dfA = dict_out['dfAvg']; dfF = dict_out['dfFull']
# dfA[dfA['NAME'].isin(c)]
#
# #              NAME  SourceXWeight  useWeightValue  AvgTargetValue
# # 2       Argentina      21.977157             1.0       21.977157
# # 6          Belize      15.000000             1.0       15.000000
# # 8         Bolivia      15.201842             1.0       15.201842
# # 12         Canada      37.983313             1.0       37.983313
# # 14          Chile      17.655511             1.0       17.655511
# # 54  United States      24.247515             1.0       24.247515
#
# # ------------------------------------------------------
# # Case 2 - weighted average -----------------------
# # provide the weight value....
# # This matches sheet 'Pivot_Full_Likelihood' in the excel file
#
# dict_out = weightedaverage(dframe=dframe,
#                            target_id='NAME', source_value='FF_Value',
#                            weight_value='Val_Like',
#                            drop_areas_with_no_source=False, verbose=False)
#
# dfA = dict_out['dfAvg']; dfF = dict_out['dfFull']
# dfA[dfA['NAME'].isin(c)]
#
# #              NAME  SourceXWeight  useWeightValue  AvgTargetValue
# # 2       Argentina           0.00            0.00             NaN
# # 6          Belize           1.05            0.07       15.000000
# # 8         Bolivia          27.84            1.42       19.605634
# # 12         Canada          25.37            0.74       34.283784
# # 14          Chile           0.00            0.00             NaN
# # 54  United States          85.15            7.64       11.145288
#
#
# # ------------------------------------------------------
# # Case 2b - weighted average, with area weighting for areas missing weights ---------
# # provide the weight value and set boolean to True.
# # Now, there are col_values for Argentina and Chile, and they match the area-weighting
#
# dict_out = weightedaverage(dframe=dframe,
#                            target_id='NAME', source_value='FF_Value',
#                            weight_value='Val_Like',
#                            target_area='akm2_targ', intersect_area='akm2_isct',
#                            area_weight_where_no_weight=True,
#                            drop_areas_with_no_source=False, verbose=False)
#
# dfA = dict_out['dfAvg']; dfF = dict_out['dfFull']
# dfA[dfA['NAME'].isin(c)]
#
# #              NAME  SourceXWeight  ...  Flag_AreaWeightAdded  AvgTargetValue
# # 2       Argentina      21.977157  ...                  True       21.977157
# # 6          Belize       1.050000  ...                 False       15.000000
# # 8         Bolivia      27.840000  ...                 False       19.605634
# # 12         Canada      25.370000  ...                 False       34.283784
# # 14          Chile      17.655511  ...                  True       17.655511
# # 54  United States      85.150000  ...                 False       11.145288
#
#
# # ------------------------------------------------------
# # Case X - weighted average, with extensive weighting, with supplemental weight when missing
# # for extensive weighting, we also have to provide the weight areas...
#
# dict_out = weightedaverage(dframe=dframe,
#                            target_id='NAME', source_value='FF_Value',
#                            weight_value='Val_Like', weight_area='akm2_like',
#                            target_area='akm2_targ', intersect_area='akm2_isct',
#                            weight_intensive=False,
#                            drop_areas_with_no_source=False, verbose=False)
#
# dfA = dict_out['dfAvg']; dfF = dict_out['dfFull']
# dfA[dfA['NAME'].isin(c)]
#
# #              NAME  SourceXWeight  useWeightValue  AvgTargetValue
# # 2       Argentina       0.000000        0.000000             NaN
# # 6          Belize       0.022526        0.001502       15.000000
# # 8         Bolivia       3.024436        0.217703       13.892491
# # 12         Canada       1.300967        0.038132       34.117805
# # 14          Chile       0.000000        0.000000             NaN
# # 54  United States      28.849045        2.785679       10.356197
#
# # ------------------------------------------------------
# # Case Xb - weighted average, with extensive weighting, with supplemental weight when missing
# # for extensive weighting, we also have to provide the weight areas...
#
# dict_out = weightedaverage(dframe=dframe,
#                            target_id='NAME', source_value='FF_Value',
#                            weight_value='Val_Like', weight_area='akm2_like',
#                            target_area='akm2_targ', intersect_area='akm2_isct',
#                            area_weight_where_no_weight=True,
#                            weight_intensive=False,
#                            drop_areas_with_no_source=False, verbose=False)
#
# dfA = dict_out['dfAvg']; dfF = dict_out['dfFull']
# dfA[dfA['NAME'].isin(c)]
#
# #              NAME  SourceXWeight  ...  Flag_AreaWeightAdded  AvgTargetValue
# # 2       Argentina      21.977157  ...                  True       21.977157
# # 6          Belize       0.022526  ...                 False       15.000000
# # 8         Bolivia       3.024436  ...                 False       13.892491
# # 12         Canada       1.300967  ...                 False       34.117805
# # 14          Chile      17.655511  ...                  True       17.655511
# # 54  United States      28.849045  ...                 False       10.356197
# endregion

