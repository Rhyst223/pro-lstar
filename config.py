# -*- coding: utf-8 -*-
import configparser
import os
import pro_lstar

from pathlib import Path

"""
Created on Fri May 13 08:23:54 2022

@author: krmurph1
"""


def get_config_file():
    """Return location of configuration file

    In order
    1. ~/.magpy/gmagrc
    2. The installation folder of gmag

    Returns
    -------
    loc : string
        File path of the gmagrc configuration file

    References
    ----------
    modeled completely from helipy/util/config.py
    """
    config_filename = 'pro_lstar_rc'

    # Get user configuration location
    home_dir = Path.home()
    config_file_1 = home_dir / 'pro-lstar' / config_filename

    module_dir = Path(pro_lstar.__file__)
    config_file_2 = module_dir / '..' / config_filename
    config_file_2 = config_file_2.resolve()

    print(type(module_dir))
    print(type(config_file_1))
    print(type(config_file_2))

    for f in [config_file_1, config_file_2]:
        if f.is_file():
            return str(f)    

def load_config():
    """Read in configuration file neccessary for loading the data
    locally

    Returns
    -------
    config_dic : dict
        Dictionary containing all options from configuration file.
    """
    
    config_path = get_config_file()
    configf = configparser.ConfigParser()
    configf.read(config_path)
    
    config_dic = {}

    data_dir = os.path.expanduser(configf['DEFAULT']['data_dir'])

    #modify directory for windows
    if os.name == 'nt':
        data_dir = data_dir.replace('/', '\\')
    config_dic['data_dir'] = data_dir

    
    return config_dic

a = get_config_file()
b = load_config()