
import pandas as pd




def return_mask_column_conditions(df, col_names, col_values):
    """
    Get a mask for ROWS from multiple equalities across columns in a dataframe
    :param df: the dataframe
    :param col_names: list of  column names, ordered to match...
    :param col_values: list of the col_values we look for in the columns
    :return: a mask (numpy array))
    TODO: should check thet input is lists, not just a single string, e.g.
    credit to # https://stackoverflow.com/questions/33699886/filtering-dataframes-in-pandas-use-a-list-of-conditions
    """
    conditions = zip(col_names, col_values)
    comparisons = [df[name] == value for name,value in conditions]
    # start with first mask
    result = comparisons[0]
    # and add on
    for comp in comparisons[1:]:
        result = result & comp
    return result


def return_mask_index_conditions(df, index_names, index_values):
    """
    Get a mask for ROWS across multiple conditions on indices (presumably multiiindex)
    ... as above
    :param df:
    :param index_names:
    :param index_values:
    :return: a mask (numpy array)
    """
    # for a multiindex (rows) with col_values to match, return the mask for the entire df

    conditions = zip(index_names, index_values)
    comparisons = [df.index.get_level_values(name) == value
                   for name, value in conditions]

    result = comparisons[0]

    for comp in comparisons[1:]:
        result = result & comp

    return result

    # create an index slice


def return_mask_index_and_col_conditions(df, index_names, index_values,
                                         col_names, col_values):
    """
    return a boolean ROW mask for a df, based on index and column conditions
    See called functions
    """
    index_mask = return_mask_index_conditions(df=df, index_names=index_names,
                                              index_values=index_values)
    col_mask = return_mask_column_conditions(df=df, col_names=col_names,
                                             col_values=col_values)

    return index_mask & col_mask

def replace_in_column(col, mask_for, new_val):

    if mask_for == 'None':
        mask = col.isnull()
    else:
        mask = (col == mask_for)
    col[mask] = new_val
    return col


