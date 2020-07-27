# important to install geopandas current:
# conda install --channel conda-forge geopandas
# per https://geopandas.org/install.html
# and may have to re-install rtree (a dependency) after that


# note that we can have issues with overprecise coordinates OR with invalid geometries

# e.g., this error with intersect: TopologyException: found non-noded intersection between
#   LINESTRING (8.76232e+06 3.3143e+06, ....  and LINESTRING (...) at
#   8762944.178470498 3312455.5381205333
# see discussion here:
#   https://gis.stackexchange.com/questions/217789/geopandas-shapely-spatial-difference-topologyexception-no-outgoing-diredge-f
#   https://gis.stackexchange.com/questions/50399/fixing-non-noded-intersection-problem-using-postgis
#   https://gis.stackexchange.com/questions/188622/rounding-all-coordinates-in-shapely
#   https://stackoverflow.com/questions/42025349/how-to-convert-polygon-that-touches-itself-in-a-line-into-valid-polygon

# (now make valid is incorporated into shapely v 1.8.0)


# Therefore this read_shapefile function can includes a rounding down of coordinates
#   If units are meters (which most are), two decimal places is cm,
#   rather than the current 6+ decimals, which is um or nm!
# If we reproject, then the rounding is lost, so
#   project everything, and round everything.


from calc.setup import config as cfg
from calc.setup.input import get_filepathext_fromfiles
from calc.setup.input import get_filetablecol
from calc.setup.input import get_data

from version_parser.version import Version  # pypi; use pip install

import geopandas as gpd
import pandas as pd
import numpy as np

from shapely.geometry import shape, mapping
from shapely import wkt
from shapely.validation import explain_validity


# from pyproj import CRS  # requires version > 2 something
# import pandas as pd
from gdal import osr


const_coord_precision = 0 # for shapely.wkt.dumps


# functions to check that table entries are not empty,
# and that they don't have empty strings or zeros.
# this is useful only because many of the 'x's are nested table/list calls


def checkstr(x): return x is not None and len(x) > 0


def checknum(x): return x is not None and x > 0


# https://pcjericks.github.io/py-gdalogr-cookbook/projection.html

def get_geofile_id_col_fromfile(filenameID):
    # assumed in the table of geofiles
    datarow = cfg.xltables[cfg.t_geo_files].loc[filenameID, :]
    the_id = datarow[cfg.xlcols[cfg.t_geo_files][cfg.s_id]]

    return the_id


def create_shapefile_fromdata(datanameID):
    # given a dataname ID associated with a regular file (i.e., not a shapefile),
    # read in the column of data and merge it onto an associated shapefile

    # This function should be called from create_shapefile_from_data, which will handle the projection

    filenameID, table, cur_data_col = get_filetablecol(datanameID=datanameID)

    if table != cfg.t_reg_files:
        # get the geofile directly, and also return the geofile name (the geofilenameID)
        return read_shapefile_fromdata(datanameID=datanameID), filenameID

        # raise TypeError(f'Function <create_shapefile_from_data> called with with '
        #                 f'datanameID = "{datanameID}", but this dataname is '
        #                 f'associated with a geofile, not a regular file')
        #

    # get the column of data as series
    # Note: we assume we are returning a column, not a matrix.
    # (single column is returned as nameless series)
    series_new_data = get_data(datanameID=datanameID, return_array=False)

    if not isinstance(series_new_data, pd.Series):
        raise TypeError(f'Function <create_shapefile_from_data> called with with '
                        f'datanameID = "{datanameID}", but this dataname is '
                        f'associated with a matrix, not a column.')

    series_new_data.name = cur_data_col

    # get the associated geofile
    geofile_nameID = get_assoc_geofile_from_data(datanameID=datanameID)

    # read the geofile in  (we're sending back to the calling function,
    #   but this time with geofile data)
    shapefile = read_shapefile_fromfile(geofilenameID=geofile_nameID, id_to_int=True)

    # Get the ID associated with the new geofile
    #   (this is not the index, but the ID to which we will merge)
    cur_id = get_geofile_id_col_fromfile(filenameID=geofile_nameID)

    # drop all columns except the id
    cols_to_drop = [c for c in shapefile.columns if c not in [cur_id, 'geometry']]
    shapefile.drop(columns=cols_to_drop, inplace=True)

    # merge the data into the geofile
    shapefile = shapefile.merge(right=series_new_data, how='inner',
                                              left_on=cur_id, right_index=True)

    return shapefile, geofile_nameID


def read_shapefile_fromdata(datanameID, new_id_col='', new_val_col='',
                            new_area_col='', proj_id=''):
    # we are reading from the data table
    # in most cases, this will be for aggregation, so we can pass ids and vals that
    # are then sent to the 'fromfile' version as a column renaming dict

    # Note that the data table has both regular files (e.g., excel), and geofiles.
    # For geofiles, we know the column of interest directly from the dataname.
    # For regular files, we have to get the data column and the associated geofile, and
    #   merge them together.

    filenameID, table, cur_data_col = get_filetablecol(datanameID=datanameID)

    if table == cfg.t_reg_files:

        # create a shapefile and get the associated geofile name
        shapefile, geofile_nameID = create_shapefile_fromdata(datanameID=datanameID)

        # Get the ID associated with the new geofile
        #   (this is not the index, but the ID to which we will merge)
        cur_id = get_geofile_id_col_fromfile(filenameID=geofile_nameID)

        # if given new name, rename the data column
        if new_val_col != '':
            shapefile.rename(columns={cur_data_col: new_val_col}, inplace=True)

        # rename current id if requested
        if new_id_col != '':
            shapefile.rename(columns={cur_id: new_id_col}, inplace=True)

        # if new_area_col != '':
        #     # we calculate area and name the column 'new_area_col'
        #     print(f'\tAdding area to shapefile = {geofile_nameID}, '
        #           f'associated with datanameID = {datanameID}')
        #     shapefile = add_area_to_shapefile(shapefile_in=shapefile,
        #                                       new_area=new_area_col,
        #                                       proj_id=proj_id)
        shapefile = check_for_projection_and_area(shpfile_to_check=shapefile,
                                                  new_area_col=new_area_col,
                                                  proj_id=proj_id,
                                                  geofilenameID=geofile_nameID)
        # else:
        #     # not recalculating area
        #     pass

        return shapefile

    else:  # table == cfg.t_geo_files:

        renaming = {}
        if new_id_col != '':
            # need to get the id column from the file table, and include this in the dictionary
            cur_id = get_geofile_id_col_fromfile(filenameID=filenameID)
            renaming[cur_id] = new_id_col

        if new_val_col != '':
            renaming[cur_data_col] = new_val_col

        return read_shapefile_fromfile(geofilenameID=filenameID, rename_dict=renaming,
                                       new_area_col=new_area_col,
                                       proj_id=proj_id)

# def round_geometry_np(geom, precision):
#
#     # this fails because of this: https://github.com/Toblerity/Shapely/issues/840
#     # which appears to be fixed
#
#     # https://gis.stackexchange.com/questions/217789/geopandas-shapely-spatial-difference-topologyexception-no-outgoing-diredge-f
#     geojson = mapping(geom)
#     # geojson is dict of type and coordinates; type is string, coordinates is a tuple
#     # tuple is structured as (((x1,y1),(x2,y2),...),)
#     # np.round(np.array(geojson['coordinates']), prec) returns a np.array structured as
#     # np.array([[[1,2],[3,4],[5,6]]]), with shape = (1,3,2)
#     # So we have to run the array back into a tuple
#     # could do as (tuple([tuple(cor) for cor in coord_array[0]]),)
#     # but this doesn't work for multipolygons
#     #
#     geojson['coordinates'] = np.round(np.array(geojson['coordinates']), precision)
#     return shape(geojson)


def round_geometry_wkt (geom, precision):
    """
    Given a precision, change geometry coordinates by rounding to that level
    :param geom: a geometry element from <class 'geopandas.geoseries.GeoSeries'>
    :param precision: integer; number of decimal places
    :return: a geometry object that has been rounded

    # this is a less elegant version of round_geometry_np

    e.g., start with 'POLYGON ((-170.7439000044051 -14.37555495213201, ....)
    dumps(poly,rounding_precision=1 )
    'POLYGON ((-170.7 -14.4, -170.7 ....)

    # But this is buggy; sometimes the rounding doesn't happen

    dumps(round_geometry_wkt(g[1],1))
    'POLYGON ((-5829486.5000000000000000 -504910.2000000000116415, ....
    dumps(round_geometry_wkt(g[1],2))
    'POLYGON ((-5829486.5000000000000000 -504910.2000000000116415, ....
    dumps(round_geometry_wkt(g[1],0))
    'POLYGON ((-5829487.0000000000000000 -504910.0000000000000000, ....

    https://gis.stackexchange.com/questions/368533/shapely-wkt-dumps-and-loads-does-not-always-preserve-rounding-precision

    """

    geom = wkt.loads(wkt.dumps(geom, rounding_precision=precision))
    return geom


def read_shapefile_fromfile(geofilenameID, new_id_col='', rename_dict=None,
                            new_area_col='', proj_id='', id_to_int=True):


    # we are reading from the geofile table
    file, path, ext = get_filepathext_fromfiles(filenameID=geofilenameID,
                                                table=cfg.t_geo_files)
    shapefile = gpd.read_file(filename=path + file + ext)

    # print(f'\tFunction <read_shapefile_fromfile> is rounding {geofilenameID} '
    #       f'from original:...')
    # print(shapefile['geometry'].head())
    #
    # # we could fix geomtry precision (to avoid the
    # shapefile.geometry = shapefile.geometry.apply(round_geometry_wkt, precision=1)
    #
    # print(f'\t... to this:')
    # print(shapefile['geometry'].head())

    # try to put index to integer:
    try:
        shapefile.index = shapefile.index.astype('int')
    except (TypeError, ValueError):
        # in testing, got Type and Value errors
        # if we cannot convert to int (e.g., index is strings), we leave as is.
        pass

    # get current ID
    curID = cfg.xltables[cfg.t_geo_files].loc[
        geofilenameID, cfg.xlcols[cfg.t_geo_files][cfg.s_id]]

    # if requested, try to put current ID to integer:
    if id_to_int:
        try:
            shapefile[curID] = shapefile[curID].astype('int')
        except (TypeError, ValueError):
            # in testing, got Type and Value errors.
            # if we cannot convert to int (e.g., index is strings), we leave as is.
            pass


    # TODO: to consider... check should be passed with either new_id_col or rename_dict

    if new_id_col != '':
        shapefile.rename(columns={curID: new_id_col}, inplace=True)

    if rename_dict is None:
        pass
    else:
        shapefile.rename(columns=rename_dict, inplace=True)

    shapefile = check_for_projection_and_area(shpfile_to_check=shapefile,
                                              new_area_col=new_area_col,
                                              proj_id=proj_id,
                                              geofilenameID=geofilenameID)

    return shapefile

def check_for_projection_and_area(shpfile_to_check, new_area_col, proj_id, geofilenameID=''):

    if proj_id != '':
        print(f'\t\tReprojecting "{geofilenameID}" to {proj_id}')

        proj_type, proj_text, conv_factor = get_projection(proj_id)

        shapefile = project_shapefile(shp=shpfile_to_check,
                                      projection_type=proj_type,
                                      projection_string=proj_text)
        print(f'\t\t\t... done reprojecting')
    else:
        # no reprojection
        print(f'\t\tnot reprojecting "{geofilenameID}" (if called from <create_shapefile_from_data>, may be reprojected later')
        shapefile = shpfile_to_check
        conv_factor = 1

    if len(new_area_col) > 0:
        # we have already projected, if needed
        print(f'\t\t calculating areas for "{geofilenameID}"')
        calc_area(shp=shapefile,
                  new_area_name=new_area_col,
                  conversion_factor=1)
    else:
        # not adding area
        pass

    return shapefile


# def add_area_to_shapefile(shapefile_in, new_area, proj_id = ''):
#     if proj_id != '':
#         print(f'\t\t reprojecting to {proj_id}')
#
#         proj_type, proj_text, conv_factor = get_projection(proj_id)
#
#         shapefile = project_shapefile(shp=shapefile_in,
#                                       projection_type=proj_type,
#                                       projection_string=proj_text)
#         print(f'\t\t\t... done reprojecting')
#
#
#     else:
#         # no reprojection
#         print(f'\t\tnot reprojecting to {proj_id}')
#         shapefile = shapefile_in
#         conv_factor = 1
#
#     calc_area(shp=shapefile, new_area_name=new_area, conversion_factor=conv_factor)
#
#
#     return shapefile


def project_shapefile(shp, projection_type='', projection_string=''):
    # info about going between osgeo and crs....
    # https://pyproj4.github.io/pyproj/stable/crs_compatibility.html

    # this will not match unless geopandas version (and pyproj) is high enough
    # that we can get a pyproj.CRS :
    # https://jorisvandenbossche.github.io/blog/2020/02/11/geopandas-pyproj-crs/

    newSpatialRef = osr.SpatialReference()

    if projection_type == '':
        projection_type = cfg.proj_crs_default
        projection_string = cfg.proj_s_default

    if projection_type == cfg.proj_crs_wkt:
        newSpatialRef.ImportFromWkt(projection_string)
    elif projection_type == cfg.proj_crs_code:
        if ':' in projection_string:
            code = projection_string.split(sep=':')
            code = int(code[1])
        else:
            code = projection_string
        newSpatialRef.ImportFromEPSG(code)
    elif projection_type == cfg.proj_crs_proj:
        newSpatialRef.ImportFromProj4(projection_string)
    else:
        raise TypeError(f'function <project_shapefile> called with unknown '
                        f'projection type= {projection_type}')

    # we now know the target type

    # get current type
    curSpatialRef = osr.SpatialReference()

    if shp.crs is None:
        # Need to set a crs in order to be able to reproject
        # assume geometric projection
        # WGS 1984 is common; https://spatialreference.org/ref/epsg/wgs-84/
        shp = shp.set_crs(epsg=4326) #, allow_override=True)


    if Version(gpd.__version__) < Version('v0.7.0'):
        # geopandas 0.6 returns a dict from crs, so we convert to proj4
        kludgy_crs = ' '.join(
            [f'+{k}={v}' for k, v in zip(shp.crs.keys(), shp.crs.values())])
        curSpatialRef.ImportFromProj4(kludgy_crs)
    else:
        # 0.7+ modern geopandas returns a pyproj.CRS object, which we can convert to wkt
        curSpatialRef.ImportFromWkt(shp.crs.to_wkt())

    # if newSpatialRef == curSpatialRef:
    # removed this check; now we always reproject
    if False:   # Note: because we use reprojection as the time to fix over-precision,
                #   Force all shapes to be reprojected
        print(f'\t\tNo reprojection needed ...')
        projshp = shp
    else:
        print(f'\t\tReprojecting to {projection_type}')
        # geopandas reproject
        projshp = shp.to_crs(projection_string)

        print(f'\t\t... and rounding geometry coordinates')
        # see round_geometry_wkt for discussion of why we do this
        print(wkt.dumps(projshp.geometry[0]))
        projshp.geometry = projshp.geometry.apply(round_geometry_wkt,
                                                  precision=const_coord_precision)
        print(wkt.dumps(projshp.geometry[0]))

        print(f'\t\t\t... done')

    return projshp


def calc_area(shp, new_area_name, conversion_factor):
    # 'geometry' is geopandas name

    # TODO: can we check that the shape is in the correct projection?

    # this modifies the original shp; no need to return it
    print(f'\t\t calculating area...')
    shp[new_area_name] = shp['geometry'].area * conversion_factor
    print(f'\t\t\t ... done ')


def get_projection(projnameID):
    # return type, the actual text, and a conversion factor

    datarow = cfg.xltables[cfg.t_proj].loc[projnameID, :]

    if checknum(datarow['Convert to km2']):
        conversion_factor = datarow['Convert to km2']
    else:
        raise TypeError(f'function <get_projection> called without a conversion factor '
                        f'in row with proj_NameID = {projnameID}')

    # do these sequentially, in order of preference

    if checkstr(datarow['Well-Known Text']):
        return cfg.proj_crs_wkt, datarow['Well-Known Text'], conversion_factor

    elif checkstr(datarow['Code']):
        return cfg.proj_crs_code, datarow['Code'], conversion_factor

    elif checkstr(datarow['Proj']):
        return cfg.proj_crs_proj, datarow['Proj'], conversion_factor

    else:
        raise TypeError(f'function <get_projection> called without a wkt, code, or proj '
                        f'in row with proj_NameID = {projnameID}')


def drop_duplicate_cols(df1, df2):
    # drop from df2

    print(f'\t\tChecking for duplicates in dataframes\n\t\tdf1.columns=\n'
          f'\t\t\t{df1.columns}\n'
          f'\t\tdf2.columns=\n'
          f'\t\t\t{df2.columns}')

    set_duplicates = set(df1.columns).intersection(df2.columns)

    set_duplicates.remove('geometry')

    if len(set_duplicates) > 0:
        print(f'\t\t\tdropped duplicates: {set_duplicates}')
        return df2.drop(columns=list(set_duplicates))

    else:
        print(f'\t\t\tNo duplicate columns')
        return df2


def multiintersect(list_shapes, how, new_area_col, new_area_conversion):
    """
    Intersect UP TO 4 shapefiles, which are already projected, and
    should have ids, vals, renamed as needed

    # For aggregation, we want to keep the target, and add info from the other shapes
    #   Therefore, the list must be passed with target first.
    #   This is enfored in config_aggreg.py, and in create_intersects.py

    # Note that we can run into issues with duplicated column names.
    #   https://stackoverflow.com/questions/35137952/pandas-concat-failing
    #   https://stackoverflow.com/questions/14984119/python-pandas-remove-duplicate-columns/40435354#40435354
    #   In this case, we have explicitly renamed all columns of interest,
    #   so we can drop duplicates safely (i.e., even if they contain the same info)
    :param list_shapes: list of geopandas geodataframes to intersect
    :param how: method (e.g., union)
    :param new_area_col: string name for new column into which we calculate areas
    :param new_area_conversion: a float describing conversion from
    projection measurement (e.g.) meters, to a standard area calculation (e.g., km2)
    :return: a shapefile
    """

    if len(list_shapes) < 2 or len(list_shapes) > 4:
        raise TypeError(f'Function <multiintersect> was called with fewer than 2 or '
                        f'more than 4 shapes.')

    print(f'\n-----------\n'
          f'\tFunction <multiintersect> intersecting shapes '
          f'1 ({list_shapes[0].shape[0]} rows) and 2 ({list_shapes[1].shape[0]} rows)...')

    # list_shapes[0].to_file(r'C:\temp\shape1.shp', driver='ESRI Shapefile')
    # list_shapes[1].to_file(r'C:\temp\shape2.shp', driver='ESRI Shapefile')
    # list_shapes[2].to_file(r'C:\temp\shape3.shp', driver='ESRI Shapefile')

    # Note that the two input shapes should have already had precision adjusted by
    #   round_geometry_wkt
    tempshp = gpd.overlay(list_shapes[0],
                          drop_duplicate_cols(list_shapes[0], list_shapes[1]), how=how)
    print('\t\t...done with shapes 1 and 2')

    # tempshp.to_file(r'C:\temp\tempshp12.shp', driver='ESRI Shapefile')

    if len(list_shapes) > 2:
        print(f'\n\t\t... adding shape 3, with {list_shapes[2].shape[0]} rows ...')

        print(f'\t\t... first, rounding intersect(1,2) geometry coordinates')
        # see round_geometry_wkt for discussion of why we do this

        print(wkt.dumps(tempshp.geometry[0]))
        tempshp.geometry = tempshp.geometry.apply(round_geometry_wkt,
                                                  precision=const_coord_precision)
        print(wkt.dumps(tempshp.geometry[0]))

        # tempshp.to_file(r'C:\temp\tempshp12_rounded.shp', driver='ESRI Shapefile')

        print(f'\t\t... first, checking intersect(1,2) validity')
        if not all(tempshp.is_valid):
            print(f'\t\t\tSome invalid geometries, so trying buffer0')
            tempshp = buffer0_invalid_geom(tempshp)

            # tempshp.to_file(r'C:\temp\tempshp12_rounded_valid.shp', driver='ESRI Shapefile')
        else:
            print(f'\t\t\tAll geometries valid')



        tempshp = gpd.overlay(tempshp,
                              drop_duplicate_cols(tempshp, list_shapes[2]), how=how)
        print('\t\t...done with shape 3')

    if len(list_shapes) == 4:
        print(f'\n\t\tadding shape 4, with with {list_shapes[3].shape[0]} rows ...')

        print(f'\t\t... first, rounding intersect(1,2,3) geometry coordinates')
        tempshp.geometry = tempshp.geometry.apply(round_geometry_wkt,
                                                  precision=const_coord_precision)

        print(f'\t\t... first, checking intersect(1,2,3) validity')
        if not all(tempshp.is_valid):
            print(f'\t\t\tSome invalid geometries, so trying buffer0')
            tempshp = buffer0_invalid_geom(tempshp)
        else:
            print(f'\t\t\tAll geometries valid')

        tempshp = gpd.overlay(tempshp,
                              drop_duplicate_cols(tempshp, list_shapes[3]), how=how)
        print('\t\t...done with shape 4')

    if len(new_area_col) > 0:
        if new_area_conversion > 0:
            calc_area(shp=tempshp,
                      new_area_name=new_area_col,
                      conversion_factor=new_area_conversion)
        else:
            raise KeyError(f'function <multiintersect> called with a '
                           f'new area columnn but not conversion factor.')

    print('\t\t\t...totally done with function <multiintersect>')
    return tempshp


def get_assoc_geofile_from_data(datanameID):
    # given a data name, get the associated geofile

    if datanameID is None:
        return None
    elif datanameID == '':
        return ''
    else:
        # no check that datanameID is in index, but it is data validated in excel
        datarow = cfg.xltables[cfg.t_data].loc[datanameID, :]

        # try the data column:
        dc_val = datarow[cfg.s_data_assocgeo]

        # try tho geo col
        gc_val = datarow[cfg.s_data_geofilename]

        if dc_val is None:
            return gc_val
        else:
            return dc_val


def buffer0_invalid_geom(shp):
    # https://gis.stackexchange.com/questions/217789/geopandas-shapely-spatial-difference-topologyexception-no-outgoing-diredge-f
    # https://gis.stackexchange.com/questions/344460/failing-unions-due-to-non-noded-intersections
    # https://stackoverflow.com/questions/42025349/how-to-convert-polygon-that-touches-itself-in-a-line-into-valid-polygon

    # (now make valid is incorporated into shapely v 1.8.0)

    mask_ok = shp.is_valid
    #buffered_geom = shp[~mask_ok].buffer(0)  # returns geometry geoseries
    buffered_geom = shp.buffer(0)

    if all(buffered_geom.is_valid):
        print(f'\t\tFunction <buffer0_invalid_geom> returned all valid geom.')
    else:
        print(f'\t\tFunction <buffer0_invalid_geom> returned some invalid geom')

    #shp.loc[~mask_ok,'geometry'] = buffered_geom
    shp['geometry']=buffered_geom

    return shp

