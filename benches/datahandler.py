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
            tealady=lambda: self._load_cxt_from_fcapy('tealady'),
            myocard=self._load_myocard,
            bankruptcy=self._load_bankruptcy,
        )

        self._raw_url_selector = dict(
            animal_movement='https://raw.githubusercontent.com/EgorDudyrev/FCApy/main/data/animal_movement.csv',
            mango='https://raw.githubusercontent.com/EgorDudyrev/FCApy/main/data/mango.csv',
            live_in_water='https://raw.githubusercontent.com/EgorDudyrev/FCApy/main/data/liveinwater.cxt',
            digits='https://raw.githubusercontent.com/EgorDudyrev/FCApy/main/data/digits.cxt',
            gewaesser='https://raw.githubusercontent.com/EgorDudyrev/FCApy/main/data/gewaesser.cxt',
            lattice='https://raw.githubusercontent.com/EgorDudyrev/FCApy/main/data/lattice.cxt',
            tealady='https://raw.githubusercontent.com/EgorDudyrev/FCApy/main/data/tealady.cxt',
            myocard='https://archive.ics.uci.edu/ml/machine-learning-databases/00579/MI.data',
            bankruptcy='https://archive.ics.uci.edu/ml/machine-learning-databases/00365/data.zip',
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

    def _load_myocard(self) -> TYPE_DATA_AND_META:
        data_name = 'myocard'

        url = self._raw_url_selector[data_name]
        df = pd.read_csv(request.urlopen(url), header=None, index_col=0)

        x_feats = [
            'AGE', 'SEX', 'INF_ANAM', 'STENOK_AN', 'FK_STENOK', 'IBS_POST', 'IBS_NASL', 'GB', 'SIM_GIPERT', 'DLIT_AG',
            'ZSN_A', 'nr11', 'nr01', 'nr02', 'nr03', 'nr04', 'nr07', 'nr08', 'np01', 'np04', 'np05', 'np07', 'np08',
            'np09', 'np10', 'endocr_01', 'endocr_02', 'endocr_03', 'zab_leg_01', 'zab_leg_02', 'zab_leg_03',
            'zab_leg_04', 'zab_leg_06', 'mmHg', 'mmHg', 'mmHg', 'mmHg', 'O_L_POST', 'K_SH_POST', 'MP_TP_POST',
            'SVT_POST', 'GT_POST', 'FIB_G_POST', 'ant_im', 'lat_im', 'inf_im', 'post_im', 'IM_PG_P', 'ritm_ecg_p_01',
            'ritm_ecg_p_02', 'ritm_ecg_p_04', 'ritm_ecg_p_06', 'ritm_ecg_p_07', 'ritm_ecg_p_08', 'n_r_ecg_p_01',
            'n_r_ecg_p_02', 'n_r_ecg_p_03', 'n_r_ecg_p_04', 'n_r_ecg_p_05', 'n_r_ecg_p_06', 'n_r_ecg_p_08',
            'n_r_ecg_p_09', 'n_r_ecg_p_10', 'n_p_ecg_p_01', 'n_p_ecg_p_03', 'n_p_ecg_p_04', 'n_p_ecg_p_05',
            'n_p_ecg_p_06', 'n_p_ecg_p_07', 'n_p_ecg_p_08', 'n_p_ecg_p_09', 'n_p_ecg_p_10', 'n_p_ecg_p_11',
            'n_p_ecg_p_12', 'fibr_ter_01', 'fibr_ter_02', 'fibr_ter_03', 'fibr_ter_05', 'fibr_ter_06', 'fibr_ter_07',
            'fibr_ter_08', 'GIPO_K', 'mmol/L', 'GIPER_Na', 'mmol/L', 'IU/L', 'IU/L', 'IU/L', 'L_BLOOD', 'ROE',
            'TIME_B_S', 'R_AB_1_n', 'R_AB_2_n', 'R_AB_3_n', 'NA_KB', 'NOT_NA_KB', 'LID_KB', 'NITR_S', 'NA_R_1_n',
            'NA_R_2_n', 'NA_R_3_n', 'NOT_NA_1_n', 'NOT_NA_2_n', 'NOT_NA_3_n', 'LID_S_n', 'B_BLOK_S_n','ANT_CA_S_n',
            'GEPAR_S_n', 'ASP_S_n', 'TIKL_S_n', 'TRENT_S_n'
        ]
        y_feats = [
            'FIBR_PREDS', 'PREDS_TAH', 'JELUD_TAH', 'FIBR_JELUD', 'A_V_BLOK', 'OTEK_LANC', 'RAZRIV', 'DRESSLER',
            'ZSN', 'REC_IM', 'P_IM_STEN', 'LET_IS'
        ]
        df.index.name = 'ID'
        df.columns = x_feats+y_feats
        meta = MetaData(name=data_name, url=url, x_feats=x_feats, y_feats=y_feats)
        return df, meta

    def _load_bankruptcy(self, tmp_dname: str = None) -> TYPE_DATA_AND_META:
        import os
        from zipfile import ZipFile
        from io import BytesIO
        import arff

        data_name = 'bankruptcy'
        tmp_dname = f'tmp_{data_name}' if not tmp_dname else tmp_dname

        assert not os.path.isdir(tmp_dname), f'Directory {tmp_dname} already exists.' \
                                             f' Remove this directory or specify another one via `tmp_dname` parameter'

        url = self._raw_url_selector[data_name]
        data = request.urlopen(url).read()
        zf = ZipFile(BytesIO(data))
        zf.extractall(tmp_dname)
        dfs = []
        for fname in os.listdir(tmp_dname):
            data = arff.load(open(f'{tmp_dname}/' + fname, 'r'))
            os.remove(f'{tmp_dname}/' + fname)

            df = pd.DataFrame(data['data'])
            dfs.append(df)
        os.rmdir(tmp_dname)
        df = pd.concat(dfs)
        del dfs

        x_feats = [f"X{i+1}" for i in range(df.shape[1]-1)]
        y_feats = ['is_bankrupted']
        df.columns = x_feats+y_feats
        meta = MetaData(name=data_name, url=url, x_feats=x_feats, y_feats=y_feats)

        return df, meta
