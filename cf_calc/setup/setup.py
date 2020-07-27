

from datetime import datetime

import calc.setup.config as cfg
import calc.setup.config_aggreg as cagg

from calc.setup.input import get_all_excel_tables
from calc.inout.files import make_filepathext
from calc.setup.create_sdmats import create_all_sdmats

from calc.setup.create_ffs import calculate_geotot_ffs
from calc.setup.create_intersects import create_save_all_intersects
from calc.functions.aggregation import create_df_output

def main_setup(main_excel_file,
               sdmat_file='', manual_sdmat_skip=False,
               fftot_file='', save_fftots=False, manual_fftot_skip=False,
               create_intersects=True,
               bln_normalize=False):

    # a global variable to append to outputs
    cfg.suffix_time = datetime.now().strftime("%Y-%m-%d_%H-%M")

    # read all excel tables, populate the cfg.xltables dictionary
    cfg.xltables = get_all_excel_tables(fullfile=
                                        make_filepathext(file=main_excel_file))


    if manual_sdmat_skip:
        pass
    else:
        if sdmat_file == '':
            # this function takes only a file name
            create_all_sdmats(sdmat_store_file=sdmat_file)
            # otherwise, sdmats are created on the fly

    if manual_fftot_skip:
        pass
    else:
        if fftot_file != '':
            # this function takes only a file name
            calculate_geotot_ffs(df_fftot_store_file=fftot_file,
                                 record_files=save_fftots)

    if create_intersects:
        create_save_all_intersects()

    if len(cagg.df_output) == 0:
        create_df_output(normalize=bln_normalize)


    # once fftots have been created, we can delete the sdmats
    # del cfg.d_sdmats
