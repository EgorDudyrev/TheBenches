"""This file describes DataHandler class that keeps track of all datasets and where they can be loaded from"""
from dataclasses import dataclass
from typing import Tuple, Optional, List
from urllib import request

import pandas as pd


@dataclass
class MetaData:
    name: str
    url: str
    x_feats: List[str]
    y_feats: Optional[List[str]] = None


TYPE_DATA_AND_META = Tuple[pd.DataFrame, MetaData]


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

        self._raw_url_selector = dict(
            animal_movement='https://raw.githubusercontent.com/EgorDudyrev/FCApy/main/data/animal_movement.csv',
            mango='https://raw.githubusercontent.com/EgorDudyrev/FCApy/main/data/mango.csv',
            live_in_water='https://raw.githubusercontent.com/EgorDudyrev/FCApy/main/data/liveinwater.cxt',
            digits='https://raw.githubusercontent.com/EgorDudyrev/FCApy/main/data/digits.cxt',
            gewaesser='https://raw.githubusercontent.com/EgorDudyrev/FCApy/main/data/gewaesser.cxt',
            lattice='https://raw.githubusercontent.com/EgorDudyrev/FCApy/main/data/lattice.cxt',
            tealady='https://raw.githubusercontent.com/EgorDudyrev/FCApy/main/data/tealady.cxt',
        )

    @property
    def available_datasets(self):
        return tuple(self._func_selector.keys())

    def load_dataset(self, data_name: str) -> TYPE_DATA_AND_META:
        df_meta = self._func_selector[data_name]()
        return df_meta

    def _load_csv_from_fcapy(self, data_name: str) -> TYPE_DATA_AND_META:
        url = self._raw_url_selector[data_name]
        df = pd.read_csv(url, index_col=0)

        meta = MetaData(name=data_name, url=url, x_feats=list(df.columns))
        return df, meta

    def _load_cxt_from_fcapy(self, data_name: str) -> TYPE_DATA_AND_META:
        from fcapy.context import FormalContext
        url = self._raw_url_selector[data_name]
        response = request.urlopen(url)
        df = FormalContext.read_cxt(data=response.read().decode()).to_pandas()
        meta = MetaData(name=data_name, url=url, x_feats=list(df.columns))
        return df, meta
