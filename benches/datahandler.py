"""This file describes DataHandler class that keeps track of all datasets and where they can be loaded from"""
from typing import Tuple
import pandas as pd
import os


TYPE_DATA_AND_META = Tuple[pd.DataFrame, dict]  # First is dataframe with data itself, second is its metadata


class DataHandler:
    def load_dataset(self, data_name: str) -> TYPE_DATA_AND_META:
        func_selector = {
            'animal_movement': self._load_animal_movement
        }
        df_meta = func_selector[data_name]()
        return df_meta

    @staticmethod
    def _load_animal_movement() -> TYPE_DATA_AND_META:
        raw_url = 'https://raw.githubusercontent.com/EgorDudyrev/FCApy/main/data/animal_movement.csv'
        df = pd.read_csv(raw_url, index_col=0)
        return df, {}
