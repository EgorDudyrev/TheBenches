"""This file describes DataHandler class that keeps track of all datasets and where they can be loaded from"""
from typing import Tuple
from urllib import request

import pandas as pd
import os


TYPE_DATA_AND_META = Tuple[pd.DataFrame, dict]  # First is dataframe with data itself, second is its metadata



class DataHandler:
    def __init__(self):
        self._func_selector = dict(
            animal_movement=lambda: self._load_csv_from_fcapy('animal_movement'),
            mango=lambda: self._load_csv_from_fcapy('mango'),
            live_in_water=lambda: self._load_cxt_from_fcapy('live_in_water'),
            digits=lambda: self._load_cxt_from_fcapy('digits'),
            gewaesser=lambda: self._load_cxt_from_fcapy('gewaesser'),
            lattice=lambda: self._load_cxt_from_fcapy('lattice'),
            tealady=lambda: self._load_cxt_from_fcapy('tealady')
        )

    @property
    def available_datasets(self):
        return tuple(self._func_selector.keys())

    def load_dataset(self, data_name: str) -> TYPE_DATA_AND_META:
        df_meta = self._func_selector[data_name]()
        return df_meta

    @staticmethod
    def _load_csv_from_fcapy(data_name: str) -> TYPE_DATA_AND_META:
        raw_url_selector = {
            'animal_movement': 'https://raw.githubusercontent.com/EgorDudyrev/FCApy/main/data/animal_movement.csv',
            'mango': 'https://raw.githubusercontent.com/EgorDudyrev/FCApy/main/data/mango.csv',
        }
        df = pd.read_csv(raw_url_selector[data_name], index_col=0)
        return df, {}

    @staticmethod
    def _load_cxt_from_fcapy(data_name: str) -> TYPE_DATA_AND_META:
        from fcapy.context import FormalContext

        raw_url_selector = {
            'live_in_water': 'https://raw.githubusercontent.com/EgorDudyrev/FCApy/main/data/liveinwater.cxt',
            'digits': 'https://raw.githubusercontent.com/EgorDudyrev/FCApy/main/data/digits.cxt',
            'gewaesser': 'https://raw.githubusercontent.com/EgorDudyrev/FCApy/main/data/gewaesser.cxt',
            'lattice': 'https://raw.githubusercontent.com/EgorDudyrev/FCApy/main/data/lattice.cxt',
            'tealady': 'https://raw.githubusercontent.com/EgorDudyrev/FCApy/main/data/tealady.cxt',
        }
        response = request.urlopen(raw_url_selector[data_name])
        df = FormalContext.read_cxt(data=response.read().decode()).to_pandas()
        return df, {}
