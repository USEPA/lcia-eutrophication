import sys

import pandas as pd


def remove_string_from_array(nparray, str_to_remove):

    # a string column extracted from a dataframe should return dtype 'object'
    # a np array created from strings should have a 'U' in the dtype
    # see https://numpy.org/doc/stable/reference/arrays.dtypes.html

    if nparray.dtype == 'object' or 'U' in str(nparray.dtype):
        arr = nparray.copy()
        return arr[arr != str_to_remove]
    else:
        return nparray


def categorize_dataframe(df, fraction_limit=0.2):
    """
    Convert columns of a data frame to dtype 'category'
    if the number of unique elements is below a threshold
    :param df: data frame
    :param fraction_limit: threshold fraction (e.g., 0.25)
    :return: the data frame

    TODO: at some point, I think the 'object' dtype will become 'string'

    """
    for col in df.columns:  # could also use iteritems?
        # print('col name={0}, type={1}, nunique={2}, and frac unique={3:5.2}'.format(
        #     col, dfisect[col].dtype, dfisect[col].nunique(),
        #     dfisect[col].nunique()/len(dfisect)
        # ))

        if (df[col].dtype == 'object') & (df[col].nunique()/len(df) < fraction_limit):
            # print('\tgoing to convert to category')
            df[col] = df[col].astype('category')

    return df



def get_size(obj, seen=None):
    # get full size of variable objects
    # https://goshippo.com/blog/measure-real-size-any-python-object/

    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size