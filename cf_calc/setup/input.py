# set up calculation inputs

# define functions to read from the various excel tables
# Note that all column names are set in config.py

# read in all the tables (we will not use all)
# set up dictionaries to hold the tables and their headers.
import os  # check for file existence, use os.sep
from collections import OrderedDict

import geopandas as gpd
import numpy as np

import openpyxl  # table functionality requires v3.0.4 (or higher)


import pandas as pd
import scipy
from scipy import io

from calc.setup import config as cfg
from calc.inout.files import add_full_directory


# print('__file__={0:<35} | __name__={1:<20} |'
#       ' __package__={2:<20}'.format(__file__, __name__,
#                                     str(__package__)))


def checkstr(x): return x is not None and len(x) > 0





def read_excel_named_table(sheet, tablename,
                           firstColIntoIndex=True, firstRowIntoColumns=True):
    """
    Read a named table (different than a named range) from excel, and return as data frame
    :param sheet: an openpyxl worksheet object
    :param tablename: a string that is the table name
    :param firstColIntoIndex: boolean,
    do we convert the first column into the dataframe index?
    :param firstRowIntoColumns: boolean,
    do we convert the first row into the datafrom columns?
    :return: pandas dataframe; any row with all nas is dropped

    note: requires openpyxl v. 3.0.4 or higher
    """
    region = sheet.tables[tablename].ref

    if firstRowIntoColumns:
        df = pd.DataFrame(([cell.value for cell in row] for row in sheet[region][1:]),
                          columns=([cell.value for cell in sheet[region][0]]))
    else:
        df = pd.DataFrame(([cell.value for cell in row] for row in sheet[region][1:]))

    if firstColIntoIndex:
        df.set_index(df.columns[0], inplace=True)

    df.dropna(axis=0, how='all', inplace=True)

    return df


def get_all_excel_tables(fullfile):
    """
    Read excel CF file and get all the tables, return them as a dictionary
    :param fullfile: path to excel file with CalcCF info.
    :return: dictionary of tables; if there is a replacement name defined,
    this will be the key; otherwise, the excel table name is the key
    """

    wb = openpyxl.load_workbook(filename=fullfile, data_only=True)
    # print(wb.worksheets)

    dict_tables = {}

    for ws in wb.worksheets:
        for table in ws.tables.items():
            # table returns a tuple of name and address

            # if we have defined a replacement, use it:
            try:
                key_name = cfg.replace_table_names[table[0]]
            except KeyError:
                # will key keyerror if the table name is not in replace_table_names
                key_name = table[0]

            dict_tables[key_name] = read_excel_named_table(sheet=ws, tablename=table[0],
                                                           firstColIntoIndex=True,
                                                           firstRowIntoColumns=True)
            # if key_name == cfg.t_calc:
            #   could do specific cleaning here


    # print(dict_tables.keys())
    return dict_tables

def check_table_item_valid(item_from_table):
    return not item_from_table is None and len(item_from_table)>0


def get_widetable_cols(index_item, wide_table):
    """
    Return information from a 'wide' format table;
    for a given row, return the column names with either T/F or numbers

    We have a handful of tables in wide format
    (index and columns are each index col_values from some other tables),
    with EITHER ordered or boolean col_values when the column is associated with the index_item
    :param index_item: an index (row) to the table
    :param wide_table: the dataframe; index(row) and columns are
    named with variables of interest
    :return: list of columns associated with the row (if ordered, in order)
    """


    # clean up the table
    # first, drop rows and columns that are totally empty
    temptable = wide_table.dropna(axis=0, how='all')
    temptable = temptable.dropna(axis=1, how='all')

    # convert None to False

    # figure out type
    # we look for any occurrences of the number 2.
    # The first .any() returns columns with True/False, and then we in that series.
    # Since 1s can be interpreted as Trues,
    # we are concerned with any ordering that goes above 1.
    bln_has_order = temptable.isin([2]).any().any()

    # get the row (comes back as a series)
    row_in_table = temptable.loc[index_item, :]

    # drop na col_values - note that this also drops None (the value we get with empty excel cells)
    row_in_table = row_in_table.dropna()

    if bln_has_order:
        # sort by number
        row_in_table = row_in_table.sort_values(ascending=True)
        list_columns = list(row_in_table.index)
    else:
        # just looking for true/false
        # select items from index (the excel columns) where value is true
        list_columns = list(row_in_table.index[row_in_table])
    return list_columns


def get_filetablecol(datanameID):
    """
    Based on a nameID from data table, get the file name and file table (sheet or geo)
    Also return the column from the data table, since there's a bit of logic in
    figuring out if this is a geofile or regular file
    :param datanameID: a string that matches something in the index of t_data
    :return: tuple (name, table, col)
    """
    # no check that there's no match in either table (or that there's a match in both)

    # get name of file
    # check in index
    if datanameID in cfg.xltables[cfg.t_data].index:

        # set empty variables
        filenameID = ''
        filetable = ''
        col_in_filetable = ''

        # look in both tables
        # file name
        regname = cfg.xltables[cfg.t_data].loc[
            datanameID, cfg.s_data_regfilename]
        # geofile name
        geoname = cfg.xltables[cfg.t_data].loc[
            datanameID, cfg.s_data_geofilename]

        # use the one that's not empty to get file name
        if regname is None:
            # not here
            pass
        elif len(regname) == 0:
            # not here
            pass
        else:
            # in the files tables, so we expect a column, or blank for a matrix
            filenameID = regname
            filetable = cfg.t_reg_files
            col_in_filetable = cfg.xltables[cfg.t_data].loc[datanameID, cfg.s_data_colregfile]

        if geoname is None:
            # not here (nothing in the cell)
            pass
        elif len(geoname) == 0:
            # not here (empty string in the cell)
            pass
        else:
            # in the files tables
            filenameID = geoname
            filetable = cfg.t_geo_files
            col_in_filetable = cfg.xltables[cfg.t_data].loc[datanameID, cfg.s_data_colgeofile]

        return filenameID, filetable, col_in_filetable

    else:
        raise KeyError(f'datanameID = "{datanameID}" was NOT in table Tbl_Data')


def get_filerow(datanameID):
    # given a datanameID, return the entire row from the corresponding table

    # get the row name and the table
    filenameID, filetable, _ = get_filetablecol(datanameID)

    return cfg.xltables[filetable].loc[filenameID, :]


def get_filepathext_fromfiles(filenameID, table):
    """
    Return tuple of directory/path information based on a filenameID
    in a given table (either the data or the gis files).
    For C:\directory\example.csv, these are the file name (just 'example'),
    the path ('C:\directory'), and extension ('.csv')
    :param filenameID: first column (the row index) in the data or gis tables
    :param table: table name (either for data or gis)
    :return: tuple of file, path, extension
    """
    if filenameID in cfg.xltables[table].index:
        datarow = cfg.xltables[table].loc[filenameID, :]

        # gotta check for trailing separators in the path and subfolder
        path = add_full_directory(datarow[cfg.xlcols[table][cfg.s_basedir]])
        if path[-1] != os.sep:
            path = path + os.sep

        folderpath = datarow[cfg.xlcols[table][cfg.s_folder]]

        # folder may be empty; if it's not, add and check the trailing separator
        if not folderpath is None:
            if len(folderpath) > 0:
                path = path + folderpath
                if path[-1] != os.sep:
                    path = path + os.sep

        fileshortname = datarow[cfg.xlcols[table][cfg.s_filename]]
        ext = datarow[cfg.xlcols[table][cfg.s_ext]]

        return fileshortname, path, ext
    else:
        raise KeyError(f'File_NameID = {filenameID} was NOT in table {table}')


def get_filepathext(datanameID):
    # given a datanameID in t_data, get the file, path, and extension

    filenameID, table, _ = get_filetablecol(datanameID)

    fileshortname, path, ext = get_filepathext_fromfiles(filenameID, table)

    return fileshortname, path, ext


def getput_idsnames_fromfiles(filenameID, table, return_dict=False):
    """
    Pull out ids and names from the file table, and write ids and names into
    appropriate columns in the file tables
    We store ids, names as dictionaries...(to avoid errors putting dataframes or series into dataframes)
    # note we must use ordered dict:
    # https://stackoverflow.com/questions/18996714/how-to-turn-pandas-dataframe-row-into-ordereddict-fast

    # but there is an option to return either the series/dataframe or dict
    :param filenameID: row index in one of the data tables
    :param table: name of the data table
    :param return_dict: boolean; return to calling function a dataframe, unless this is true
    :return: ids as series, names as dataframe (index is the ids)
    """


    # we should only get names and IDs from geofiles...
    if table == cfg.t_geo_files:
        # we assume that ONLY geofiles have the full list of names
        # so we set a boolean to allow recording of ids
        bln_record_ids = True
    else:
        # we will not record the ids into the geofiles table, but will just return what we read
        bln_record_ids = False

    if filenameID in cfg.xltables[table].index:

        # first we check if names and IDs are already here
        datarow = cfg.xltables[table].loc[filenameID, :]

        if bln_record_ids:
            if all(x in datarow.index for x in [cfg.add_geo_id, cfg.add_geo_name]):
                # we already have the id and name columns, so there may be data
                pass
            else:
                # there are no columns with the id and names, so we add them
                cfg.xltables[table][cfg.add_geo_id] = ''
                cfg.xltables[table][cfg.add_geo_name] = ''

                # re-retrieve the data row
                datarow = cfg.xltables[table].loc[filenameID, :]

            # now, we know that the data table contains ids and names
            # check if there are lists, dictionaries, etc... but NOT empty strings
            bln_have_ids = datarow[cfg.add_geo_id] != ''
            bln_have_names = datarow[cfg.add_geo_name] != ''

        else:
            # bln_record_ids is false, so we get data anew
            bln_have_ids = False
            bln_have_names = False

        # now we have determined whether we already have names/ids in the geofile table
        # (Note: in the case of looking up a file in the regular file table,
        #   we DO NOT get ids from the geofile table, but rather read from the new file)

        if not bln_have_ids or not bln_have_names:
            # one is missing, so we need to get the file
            filename, path, ext = get_filepathext_fromfiles(filenameID, table)

            if not os.path.isfile(path + filename + ext):
                raise FileExistsError(f'Function oldget_data_from_datanameID '
                                      f'did not find file {path + filename + ext}, '
                                      f'and {filenameID} in table {table}')

            col_id = datarow[cfg.xlcols[table][cfg.s_id]]

            # for the geofiles table, there will be a name column, but not necessarily a value
            # for regular files, there is NOT a name column
            if table == cfg.t_geo_files:
                # this is a bit kludgy... this function was originally for just the
                # table geofiles, but I expanded it to include both tables.
                 if cfg.s_geofile_namecol in datarow.index:
                     col_name = datarow[cfg.s_geofile_namecol]
            else:
                col_name = None

            if col_name is None:
                bln_have_name_header = False
            else:
                if len(col_name) > 0:
                    bln_have_name_header = True
                else:
                    bln_have_name_header = False

            # we'll need the ids whether or not we're missing ids or names
            # TODO: ids and names could return numbers represented as strings...
            # e.g., the GeosGrid from Roy.  Do we force conversion to number?

            if table == cfg.t_reg_files:
                # need to set a sheet name
                xlsheet = datarow[cfg.s_regfile_xlsheet]
            else:
                xlsheet = ''

            ids = read_column_from_file(path=path, file=filename, extension=ext,
                                        col=col_id, sheet=xlsheet,
                                        return_array=False)
            try:
                ids = ids.astype(int)
            except ValueError:
                # if we cannot convert to int, we get a value error
                pass

            if bln_have_name_header and not bln_have_names:
                # there is a name field, but we don't have names yet
                names = read_column_from_file(path=path, file=filename,
                                              extension=ext, col=col_name)

                # join the ids (as index) to the name col_values
                df_names = pd.DataFrame(data=names.values, index=ids.values)

            if bln_record_ids:
                # write to cfg.xltables

                # write the ids if we don't have them
                if not bln_have_ids:
                    cfg.xltables[table].loc[filenameID, cfg.add_geo_id] = \
                        [ids.to_dict(into=OrderedDict)]


                if bln_have_name_header:
                    if not bln_have_names:
                        cfg.xltables[table].loc[filenameID, cfg.add_geo_name] = \
                            [df_names.to_dict(into=OrderedDict)]
                    else:
                        pass
                        # names are already there, and we didn't read them, anyhow.
                else:
                    cfg.xltables[table].loc[filenameID,
                                            cfg.add_geo_name] = 'No Name'

        # now we either had the IDS and names, or we have created them...
        # These are dictionaries... though the name might be 'no name'
        # so we can return:
        if not bln_record_ids:
            return ids
        else:

            # get them back out of the table
            ids = cfg.xltables[table].loc[filenameID, cfg.add_geo_id]
            names = cfg.xltables[table].loc[filenameID, cfg.add_geo_name]

            if return_dict:
                return ids, names
            else:
                if names == 'No Name':
                    return pd.Series(ids), names
                else:
                    # we have both ids and names that can be converted to series and dict
                    return pd.Series(ids), pd.DataFrame.from_dict(names,
                                                                  orient='columns')

    else:
        raise KeyError(f'function get_ids_names_fromfiles '
                       f'did not find file name = {filenameID} '
                       f'in index of table {table}')


def getput_idsnames(datanameID, return_dict=False):
    # figure out which geofilenameID and table we're working in
    # geofilenameID is the unique index value in one of the
    # two data tables

    # returns ids and names (either as dfs or dicts)

    filenameID, table, _ = get_filetablecol(datanameID)

    # we can return the output of the file names directly
    return getput_idsnames_fromfiles(filenameID=filenameID,
                                     table=table,
                                     return_dict=return_dict)


def get_data(datanameID, return_array=False):
    """
    # given a datanameID , pull out the data,
    # returning as a numpy array if so specified
    :param datanameID: the row index in the table of data
    :param return_array: boolean; if true, return as numpy array
    :return: a series (with ids) if a column, or numpy array (no ids) of
    the data associated with datanameID
    """


    print(f'\tFunction <get_data> called with return_array={return_array}')

    # get_filetablecol tells us the corresponding source file nameID and the
    #  appropriate table (either sheetfiles or geofiles)
    # e.g., for a Tbl_Data entry of FFs, we need to know which excel file to look in
    filenameID, table, col = get_filetablecol(datanameID=datanameID)

    if filenameID in cfg.xltables[table].index:
        filename, path, ext = get_filepathext(datanameID=datanameID)

        if not os.path.isfile(path + filename + ext):
            raise FileExistsError(f'Function get_data '
                                  f'did not find file {path + filename + ext}, '
                                  f'\n\tassociated with {datanameID} in datatable '
                                  f'and {filenameID} in table {table}')

        tblrow = cfg.xltables[table].loc[filenameID, :]

        if table == cfg.t_reg_files:
            # check if is matrix
            # to handle cell being blank or not being set to true...

            if not tblrow[cfg.s_regfile_ismatrix] is None and \
                    tblrow[cfg.s_regfile_ismatrix]:
                # this is a matrix
                # check if there is a matrix, based on whether we have associated geo file
                associatedGeoFile = tblrow[cfg.s_regfile_assocgeo]
                matlabName = tblrow[cfg.s_regfile_matname]
                hasids = tblrow[cfg.s_regfile_matrixhasIDs]

                return read_matrix_from_file(path=path, file=filename,
                                             extension=ext,
                                             matrix_has_ids=hasids,
                                             geofile=associatedGeoFile,
                                             matlabName=matlabName,
                                             return_array=return_array)

            # there is no matrix, so this is a standard file retrieval
            thesheet = tblrow[cfg.s_regfile_xlsheet]
            theColID = tblrow[cfg.xlcols[cfg.t_reg_files][cfg.s_id]]

            # but we may need to fix the ids,
            # because many datafiles may lack data for some geo ids.
            # So get the preliminary data

            tempdata = read_column_from_file(path=path,
                                             file=filename,
                                             extension=ext,
                                             sheet=thesheet,
                                             col=col,
                                             colID=theColID)

            # sometimes there are n/a, or NA(), etc. col_values in excel files,
            #    so we replace with zero
            tempdata.fillna(0, inplace=True)

            # and then send it through the id fixer
            return fix_data_ids(df=tempdata,
                                ref_geofile=cfg.xltables[cfg.t_data].loc[
                                    datanameID, cfg.s_data_assocgeo],
                                sort=True)



        else:
            # in the geofiles table
            thesheet = ''
            theColID = tblrow[cfg.xlcols[cfg.t_geo_files][cfg.s_id]]

            # now we have gotten relevant info
            return read_column_from_file(path=path, file=filename, extension=ext,
                                         sheet=thesheet,
                                         col=col,
                                         colID=theColID)

    else:
        raise KeyError(f'In function <get_data>, '
                       f'called with datanameID = {datanameID},'
                       f'\n\tfile name = {filenameID} not found in '
                       f'index of table {table}')


def get_data_fromfiles(filenameID, return_array=False):
    """
    reads an entire excel sheet or matrix.
    HasIDs is outside of the 'is matrix' check,
    because we may want to assume IDs are on an excel sheet.
    :param filenameID: name of file (row) from ONLY the regular files table
    :param return_array: boolean; return numpy array if true
    :return: data as column or matrix, as called.
    """
    # we know table is cfg.t_reg_files
    # this is for reading the entire sheet or entire matrix.

    if filenameID in cfg.xltables[cfg.t_reg_files].index:
        # get file inpfo so we can read
        filename, path, ext = get_filepathext_fromfiles(filenameID=filenameID,
                                                        table=cfg.t_reg_files)

        if not os.path.isfile(path + filename + ext):
            raise FileExistsError(f'Function <get_data> '
                                  f'did not find file {path + filename + ext}, '
                                  f'\n\tassociated with {filenameID} '
                                  f'in table {cfg.t_reg_files}')

        tblrow = cfg.xltables[cfg.t_reg_files].loc[filenameID, :]
        hasids = tblrow[cfg.s_regfile_matrixhasIDs]

        # to handle cell being blank or not being set to true...
        if not tblrow[cfg.s_regfile_ismatrix] is None and \
                tblrow[cfg.s_regfile_ismatrix]:

            # this is a matrix
            # check if there associated geo file
            associatedGeoFile = tblrow[cfg.s_regfile_assocgeo]
            matlabName = tblrow[cfg.s_regfile_matname]

            return read_matrix_from_file(path=path, file=filename,
                                         extension=ext,
                                         matrix_has_ids=hasids,
                                         geofile=associatedGeoFile,
                                         matlabName=matlabName,
                                         return_array=return_array)

        else:
            thesheet = tblrow[cfg.s_regfile_xlsheet]

            return read_matrix_from_file(path=path, file=filename,
                                         extension=ext,
                                         matrix_has_ids=hasids,
                                         xlsheet=thesheet,
                                         return_array=return_array)

    else:
        raise KeyError(f'In function <get_data_fromfiles>, '
                       f'\n\tfile name = {filenameID} not found in '
                       f'index of table {cfg.t_reg_files}')


def fix_data_ids(df, ref_geofile, sort=True):
    """
    based on ids in reference geofile, reindex and sort the dataframe
    :param df: dataframe to be reindexed
    :param ref_geofile: name of geofile (in the table of geofiles) with the 'authoritative' list of ids
    :param sort: boolean: do we sort the indices
    :return: reindexed, and possibly sorted, dataframe
    """

    ids, _ = getput_idsnames_fromfiles(filenameID=ref_geofile,
                                       table=cfg.t_geo_files,
                                       return_dict=False)

    if isinstance(df, pd.Series):
        bln_fix_cols = False
    elif isinstance(df, pd.DataFrame):
        # assume that we can reindex a square matrix...
        # this could be problematic in cases with few IDs and lots of other columns,
        # if they have to be the same number
        bln_fix_cols = df.shape[0] == df.shape[1]
    else:
        bln_fix_cols = False
        raise TypeError(f'Function <fix_data_ids> called with a '
                        f'data type ({type(df)})'
                        f'that is not pd.Series or pd.DataFrame.')

    df2 = df.reindex(index=ids, fill_value=0)

    if bln_fix_cols:
        df2 = df2.reindex(columns=ids, fill_value=0)

    if sort:
        df2.sort_index(inplace=True)
        if bln_fix_cols:
            df2.sort_index(inplace=True)

    return df2


def read_column_from_file(path, file, extension, col, sheet='',
                          colID='', return_array=False):
    """
    Read columns from xlsx, csv, mat, npy, or shp files
    We assume excel file structured with column headings.
    If data are a column, try to return a pd Series with index being the IDs

    For non-matrix files, this returns a series

    :param path:
    :param file:
    :param extension:
    :param col: column from which to get excel data; if column is empty,
    we should be reading a matrix from matlab, numpy
    :param sheet: excel sheet; if empty, we have a shapefile
    :param colID: for excel or csv, associated column with IDs corresponding to the data
    :param return_array: boolean; if true, return numpy array
    :return: column data: series with index being the IDs;
    matrix data: dataframe with index/columns being IDs
    """

    colExtract = ''

    if extension == '.shp':
        # the ID is returned as part of the geodataframe
        # use geopandas to read;
        print(f'\t\treading shapefile {file + extension}...')
        shp = gpd.read_file(path + file + extension)
        print('\t\t\t...done')
        if col in shp.columns:
            # we're extracting from a geodataframe, so we get the series
            colExtract = shp.loc[:, col]

            # not changing name of series index
        else:
            raise KeyError(f'function read_column_from_file '
                           f'could not find column {col} in '
                           f'shapefile {file} with column list: {shp.columns}')

    elif extension == '.xlsx' or extension == '.csv':
        # use pandas to read

        print(f'\t\treading excel or csv for col= {col}, '
              f'colID= {colID} (okay if colID blank)')
        if extension == '.xlsx':
            df = pd.read_excel(io=path + file + extension, sheet_name=sheet)
        else: # extension == '.csv':
            df = pd.read_csv(filepath_or_buffer=path + file + extension,
                              header=0)

        if col in df.columns:
            # note that vals is extracted as a series, so it has a (0..n) index already.
            vals = df.loc[:, col]

            # even though colID default is '', if we read an empty ID from
            # an excel sheet, then colID is None, so we check using checkstr
            if checkstr(colID):
                idx = df.loc[:, colID]
                colExtract = pd.Series(data=vals.to_numpy(), index=idx)
                # since we have indices, we sort
                colExtract.sort_index(axis=0,inplace=True)
            else:
                # we cannot sort...
                colExtract = pd.Series(data=vals)

        else:
            raise KeyError(f'Function <read_column_from_file> '
                           f'could not find column {col} '
                           f'in file = {file+extension} with column list: {df.columns}')

    # have not yet added getting IDs from mat or npy
    # elif extension == '.mat':
    #
    #     loaded = io.loadmat(path + file + extension)
    #     # print(loaded.keys())
    #     mat = loaded[matlabName]  # need to get the right variable out of the mat file
    #     return 'Not yet getting IDs'
    #
    # elif extension == '.npy':
    #     # we do not grab any IDs in this function
    #     npy = np.load(path + file + extension)
    #     return 'Not yet getting IDs'

    else:
        raise KeyError(f'function <read_column_from_file> was '
                       f'given an unknown extension: {extension}')

    # now we have defined the col extract,
    # # and we can return, based on dataframe or array
    if not isinstance(colExtract, str):
        if return_array:
            return colExtract.values
        else:
            return colExtract


def read_matrix_from_file(path, file, extension, matrix_has_ids=False,
                          geofile='', matlabName='', xlsheet='',
                          return_array=False):

    # return a matrix from the table tregfiles (table regular files)
    # if return_array = True, we return just the numbers
    # if not, we return a dataframe.
    # if matrix_has_ids is true, then the file itself has ids, and we set these as index and column
    # if matrix_has_ids is false, then we grab ids from the geofile
    # if return_array is false (we need ids), but matrix_has_ids=False and geofile='',
    #    then we return an error

    print(f'\tFunction <read_matrix_from_file> called with return_array={return_array}')

    if not return_array and not (matrix_has_ids or geofile != ''):
        raise TypeError(f'Function <read_matrix_from_file> called to return a '
                        f'dataframe, but no ids or geofile supplied.'
                        f'\tfile = "{file+extension}", in directory = {path}')

    mat = ''
    bln_got_df = False

    print(f'\t\tLoading "{file + extension}" in function read_matrix_from file...')

    if extension == '.mat':
        loaded = io.loadmat(path + file + extension)
        # print(loaded.keys())
        mat = loaded[matlabName]  # need to get the right variable out of the mat file

    elif extension == '.npy':
        mat = np.load(path + file + extension)

    elif extension == '.csv':
        # mat = pd.read_csv(filepath_or_buffer=path+file+extension,
        #                   sep=',', index_col=None, header=None)
        # mat = mat.col_values
        mat = np.genfromtxt(path + file + extension, delimiter=',')

    elif extension == '.xlsx':
        df = pd.read_excel(path + file + extension, sheet_name=xlsheet)
        bln_got_df = True
        mat = df.to_numpy()

    else:
        raise TypeError(f'function <read_matrix_from_file> called with unknown'
                        f'extension (not mat, npy, csv, or xlsx)'
                        f'\nfile = {file + extension} in directory = {path}')

    if isinstance(mat, str) and mat == '':
        # never reassigned mat... so this sould be because of unknown extension.
        # shouldn't be able to reach this point.  ha ha ha
        raise RuntimeError('In <function read_matrix_from_file>, you reached'
                           'an unreachable point.  Congrats.')
    else:
        if return_array:
            if matrix_has_ids:
                # remove first row and column
                return mat[1:, 1:]
            else:
                return mat
        else:
            if matrix_has_ids:
                if bln_got_df:
                    # already got the df
                    pass
                else:
                    row_ids = try_convert_ids_to_int(mat[1:, 0])
                    col_ids = try_convert_ids_to_int(mat[0, 1:])

                    df = pd.DataFrame(data=mat[1:, 1:],
                                    index=row_ids, columns=col_ids)
            else:
                # get geo IDs to make a dataframe

                theIDs, _ = getput_idsnames_fromfiles(filenameID=geofile,
                                                      table=cfg.t_geo_files,
                                                      return_dict=False)
                # create dataframe
                # note that we may create sparse matrices...
                # to convert: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.sparse.to_dense.html
                # there are probably more elegant/faster/more efficient ways to do this.

                if isinstance(mat, scipy.sparse.csc.csc_matrix):
                    print(f'\t\tconverting sparse matrix {file + extension} to dataframe...')
                    df = pd.DataFrame.sparse.from_spmatrix(data=mat,
                                                           index=theIDs.array,
                                                           columns=theIDs.array)
                else:
                    print(f'\t\tconverting matrix {file + extension} to dataframe...')
                    df = pd.DataFrame(data=mat, index=theIDs.array, columns=theIDs.array)
                print('\t\t\t...done')

            return df


def try_convert_ids_to_int(theids):
    # convert to the ids to integer, if we can
    try:
        theids = theids.astype(int)
    except ValueError:
        # if we cannot convert to int, we get a value error
        pass

    return theids


