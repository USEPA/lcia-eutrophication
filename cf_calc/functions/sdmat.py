import pandas as pd

from calc.functions.cleanup import remove_string_from_array

# def create_sdm_from_geo (geofile1, geofile2):
# need to pass the IDs as well.  Perhaps pass them as lists or dictionary
# {s_data_geofilename:filename, 'id_col':string}
# run intersection, and then pass the dataframe to create_sdm_from_df

print('cur version: j')

def create_sdm_from_df2(dframe, col_i, col_j, col_vals='',
                        col_i_area='', col_j_area='',
                        col_intersect_area='', divide_area_by='',
                        force1to1=False, i_values_intensive=True,
                        normalize_by_axis=''):
    """
    Create a spatial distribution matrix, used to spatially translated between
    two different geometries.  Rows(i) are 'input' and Columns(j) are output.
    :param dframe: dataframe of intersections; each row has i and j ids,
    and may contain the area of their intersection
    :param col_i: name of the column with i ids
    :param col_j:
    :param col_vals: name of the column with col_values (if not using area fractions)
    :param col_i_area: name of the column with i areas
    :param col_j_area:
    :param col_intersect_area: name of the column with intersection areas
    :param divide_area_by: do we normalize by i or j (see below for discussion of intensive/extensive)
    :param force1to1: boolean; if true, put a 1 at each intersection
    :param i_values_intensive: boolean: are the i col_values intensive?
    :param normalize_by_axis: force the max col_values per row or column to be 1
    :return: a dataframe as described above.
    """

    print(f'\tRunning function <create_sdm_from_df2>...')

    #    % SDMs.  This is an SDM where each row i, col j = is the fraction of air cell i is in lme j
    #    %     The sum along a row here are <= 1
    # need to think about what i,j means.

    # TODO: for aggfunction if col_vals are provided,
    #  could check (drop_duplicates) there are not multiple rows with i,j the same.
    #  We will have multiple rows in the case of a 3-shapefile intersect,
    #  but with just 2, this shouldn't be an issue

    df = dframe.copy()
    bln_proceed = False
    aggfunction = ''

    # note:
    # for air, this returns strings: array(['295', '296', '297', ..., '12300', '12301', '12302'], dtype=object)

    # we don't need to sort this yet; (unique doesn't sort) we do a sort at the end
    # returns numpy nd array
    ids_i = df.loc[:, col_i].dropna().unique()
    ids_j = df.loc[:, col_j].dropna().unique()

    ids_i = remove_string_from_array(nparray=ids_i, str_to_remove='')
    ids_j = remove_string_from_array(nparray=ids_j, str_to_remove='')
    # print(f'unique ids_j: {ids_j}')

    if col_vals != '':
        # we can proceed directly; ignore the col_i_areas and col_j_areas
        bln_proceed = True
        # we ASSUME that there is only one value per intersection
        aggfunction = 'sum'
    else:
        # we get into an area division (messy) situation:
        if col_vals == '':
            # we have to get col_vals through areas or setting to 1
            col_vals = 'ThisFunctionColVals'  # hard-coded; used internally in this function

            if force1to1:
                # will set all intersections to 1
                bln_proceed = True
                df[col_vals] = 1
                # note that when we do the pivot, we'll take the mean
                # in case there are multiple intersections, we just want a value of one.
                aggfunction = 'mean'

            elif col_intersect_area != '' and (col_i_area != '' or col_j_area != ''):
                # elif intersect_area is not blank, and either i_area or j_area not blank
                # Calculate area fractions first
                bln_proceed = True

                # # drop the na col_values, which should also remove empty indices
                # # i.e., na would occur in an area if
                # # there was not an intersection with the i or j in question
                # if col_i_area != '':
                #     df.dropna(subset=[col_i_area], inplace=True)
                # if col_j_area != '':
                #     df.dropna(subset=[col_j_area], inplace=True)

                # calculate the col_values as the fraction of j or i in the intersect
                # We will use SDM to calculate an overall value for polygon j,
                # so this is this is area-based weighting with intensive col_values.
                # (i.e., the col_values from col i are not adjusted based on area fractions)
                # maybe need to watch out for division by zero

                if divide_area_by == 'i':
                    df[col_vals] = df[col_intersect_area] / df[col_i_area]
                    if not i_values_intensive:
                        # cannot do area-weighting by i and also have it extensive... I think
                        bln_proceed = False

                elif divide_area_by == 'j':
                    df[col_vals] = df[col_intersect_area] / df[col_j_area]

                    if not i_values_intensive:
                        # we further reduce the col_vals by the i area fractions
                        df[col_vals] = (df[col_vals] *
                                        (df[col_intersect_area] / df[col_i_area]))
                else:
                    bln_proceed = False

                aggfunction = 'sum'

            else:
                # cannot proceed
                bln_proceed = False
                return 0

    if not bln_proceed:
        return 0
    else:
        sdm = pd.pivot_table(df, index=col_i, columns=col_j, values=col_vals,
                             aggfunc=aggfunction)

        # fill it out... even with 'dropna=False' the pivot_table drops rows with na
        sdm = sdm.reindex(index = ids_i, columns = ids_j)

        # convert to numeric...
        # we have now done this in calling functions, but we repeat here
        # sometimes data were converting to float
        try:
            sdm.index = sdm.index.astype('int')
            sdm.columns = sdm.columns.astype('int')
        except (TypeError, ValueError):
            # in testing, got Type and Value errors
            # if we cannot convert to int (e.g., index is strings)
            pass


        # sort 'em
        sdm.sort_index(axis='index', inplace=True)
        sdm.sort_index(axis='columns', inplace=True)

        sdm.fillna(value=0, inplace=True)
        # print(sdm.head())

        sdm.index.set_names(names=col_i, level=None, inplace=True)
        sdm.columns.set_names(names=col_j, level=None, inplace=True)

        if normalize_by_axis == 0:
            # make it so row sum is 1
            # (i.e., the sum will add down a column to create a 1 x n vector of 1s)
            sdm = sdm.div(sdm.sum(axis=0), axis=1)

        elif normalize_by_axis == 1:
            sdm = sdm.div(sdm.sum(axis=1), axis=0)

            # TODO from matlab version, could check that sums are < 1:
            # if bln_normalize
            #     % divide by columns of the A value sums
            #     m_ab = m_ab ./ valuesA ;
            #
            #     % check sums of rows
            #     rowSums = sum(m_ab,2) ;
            #
            #     if ~exist('tol','var')
            #         tol = 1e-6 ;
            #     end
            #
            #     if any(rowSums > 1 + tol)
            #         fprintf ('whoops; rows sums are greater than 1 + tol.  Max = %0.6f.\n', max(rowSums)) ;
            #         bln_error = true ;
            #     end
            # end
        print(f'\t\t... done with function <create_sdm_from_df2>.')
        return sdm



# region Testing
#
# # Use the toy example ------------------
# # work from "W:\work\projects\eutroCFs_EPA_ERG\Mapping\spatial distribution matrix.pptx"
# d = {'id_i':[1,1,2,3,3,3,4,4,5,5,6,6],
#      'id_j':['','A','A','','A','B','A','B','','B','B',''],
#      'a_isect':[0.6,0.4,1,0.55,0.15,0.3,0.4,0.6,0.9,0.1,0.2,0.8]}
# df_isect = pd.DataFrame(d)
#
# # all of the i areas are one
# df_area_i = pd.DataFrame({'num':list(range(1,7,1)),
#                          'area_i':[1]*6})
#
# # we calculate the j areas, keep only the A & B (ignore nans), and rename indices.
# df_area_j = df_isect.groupby('id_j').sum()
# df_area_j = df_area_j.loc[['A','B'],['a_isect']]
# df_area_j.rename(columns={'a_isect':'area_j'}, inplace = True)
# df_area_j.index.name = 'id_j'
#
# # merge them together
# df = df_isect.merge(df_area_i.rename(columns={'num':'id_i'}), how='left', on='id_i')
# df = df.merge(df_area_j, how='left', left_on='id_j', right_on='id_j')
# print(df)
#
# df_SDM_int = create_sdm_from_df2(df=df, col_i='id_i', col_j='id_j', col_j_area='area_j',
#                         divide_area_by='j', col_intersect_area='a_isect')
# print(df_SDM_int)
#
# # id_j         A         B
# # id_i
# # 1     0.205128  0.000000
# # 2     0.512821  0.000000
# # 3     0.076923  0.250000
# # 4     0.205128  0.500000
# # 5     0.000000  0.083333
# # 6     0.000000  0.166667
#
# # as shown in file:///C:/CalcDir/Python/CalcCFs/test%20sdm%20and%20aggregation_2020-07-09.html,
# # a FF matrix can be multiplied by this SDM to do area-based aggregation of an intensive FF.
#
# # I think we could also do an extensive FF (have a parameter for that in the SDM function)
#
# # Try a 1:1 dataframe --------------------------------------------------------------
# df = pd.read_csv(r'W:\work\projects\eutroCFs_EPA_ERG\Mapping\DemoAggregation\TargSourceLikelihood.csv')
# col_i = 'id_1'
# col_j = 'ISO_3DIGIT'
# fsdm1to1 = create_sdm_from_df2(df = df, col_i=col_i, col_j=col_j, force1to1=True)
# endregion

