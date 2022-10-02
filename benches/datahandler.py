"""This file describes DataHandler class that keeps track of all datasets and where they can be loaded from"""
from dataclasses import dataclass
from dataclasses_json import dataclass_json
import pkg_resources
from typing import Tuple, Optional, List, Callable
from urllib import request
from io import StringIO, BytesIO
from zipfile import ZipFile
import os
import re
import json

import pandas as pd


@dataclass_json
@dataclass
class MetaData:
    name: str
    url: str
    loading_function: str
    x_feats: List[str]
    y_feats: Optional[List[str]] = None
    classes: List[str] = None


TYPE_DATA_AND_META = Tuple[pd.DataFrame, MetaData]


class DataHandler:
    @staticmethod
    def available_datasets():
        return tuple([fname.replace('.json', '') for fname in pkg_resources.resource_listdir(__name__, 'meta_data')])

    @staticmethod
    def load_metadata(data_name: str) -> MetaData:
        stream = pkg_resources.resource_stream(__name__, f'meta_data/{data_name}.json')
        dct = json.loads(stream.read())
        dct['name'] = data_name
        return MetaData.from_dict(dct)

    def load_dataset(self, data_name: str, tmp_dname: str = None) -> TYPE_DATA_AND_META:
        current_params = locals()

        meta = self.load_metadata(data_name)
        func = self.__getattribute__(meta.loading_function)

        params = set(func.__code__.co_varnames[1:func.__code__.co_argcount])
        kwargs = {k: v for k, v in current_params.items() if k in params}
        df, meta = func(meta=meta, **kwargs)

        assert len(set(meta.x_feats)) == len(meta.x_feats)
        if meta.y_feats:
            assert len(set(meta.y_feats)) == len(meta.y_feats)

        return df, meta

    @staticmethod
    def _load_csv_from_fcapy(meta: MetaData) -> TYPE_DATA_AND_META:
        df = pd.read_csv(meta.url, index_col=0)
        meta.x_feats = list(df.columns)
        return df, meta

    @staticmethod
    def _load_cxt_from_fcapy(meta: MetaData) -> TYPE_DATA_AND_META:
        from fcapy.context import FormalContext
        response = request.urlopen(meta.url)
        df = FormalContext.read_cxt(data=response.read().decode()).to_pandas()
        meta.x_feats = list(df.columns)
        return df, meta

    @staticmethod
    def _load_myocard(meta: MetaData) -> TYPE_DATA_AND_META:
        df = pd.read_csv(meta.url, header=None, index_col=0)
        df.columns = meta.x_feats+meta.y_feats
        return df, meta

    @staticmethod
    def _load_iris(meta: MetaData) -> TYPE_DATA_AND_META:
        df = pd.read_csv(meta.url, header=None)
        df.columns = meta.x_feats+meta.y_feats
        return df, meta

    @staticmethod
    def _load_wine(meta: MetaData) -> TYPE_DATA_AND_META:
        df = pd.read_csv(meta.url, header=None)
        df.columns = meta.x_feats + meta.y_feats
        return df, meta

    @staticmethod
    def _load_haberman(meta: MetaData) -> TYPE_DATA_AND_META:
        df = pd.read_csv(meta.url, header=None)
        df.columns = meta.x_feats + meta.y_feats
        return df, meta

    @staticmethod
    def _load_ecoli(meta: MetaData) -> TYPE_DATA_AND_META:
        response = request.urlopen(meta.url)
        s = re.sub(r" +", "\t", response.read().decode())
        df = pd.read_table(StringIO(s), sep='\t', header=None)
        df.columns = meta.x_feats + meta.y_feats
        return df, meta

    @staticmethod
    def _load_breast_w(meta: MetaData) -> TYPE_DATA_AND_META:
        df = pd.read_csv(meta.url, header=None)
        df.columns = ['ID Number']+meta.y_feats+meta.x_feats
        df = df.set_index('ID Number')
        return df, meta

    @staticmethod
    def _load_spambase(meta: MetaData) -> TYPE_DATA_AND_META:
        df = pd.read_csv(meta.url, header=None)
        df.columns = meta.x_feats + meta.y_feats
        return df, meta

    @staticmethod
    def _load_parkinsons(meta: MetaData) -> TYPE_DATA_AND_META:
        df = pd.read_csv(meta.url)
        df = df.set_index('name')
        meta.x_feats = list(df.drop(meta.y_feats, axis=1).columns)
        return df, meta

    @staticmethod
    def _load_statlog(meta: MetaData) -> TYPE_DATA_AND_META:
        dfs = []
        for part, part_novowel in [('train', 'trn'), ('test', 'tst')]:
            url = meta.url.format(part_novowel=part_novowel)
            df = pd.read_csv(url, sep=' ', header=None)
            df.columns = meta.x_feats + meta.y_feats
            df['dataset_part'] = part
            dfs.append(df)

        df['class'] = [meta.classes[i - 1] for i in df['class']]
        df = pd.concat(dfs).reset_index(drop=True)

        return df, meta

    @staticmethod
    def _load_credit_card(meta: MetaData) -> TYPE_DATA_AND_META:
        df = pd.read_excel(meta.url, header=1).set_index('ID')
        meta.x_feats = list(df.drop(meta.y_feats, axis=1).columns)
        return df, meta

    @staticmethod
    def _load_sensorless(meta: MetaData) -> TYPE_DATA_AND_META:
        df = pd.read_csv(meta.url, header=None, sep=' ')
        df.columns = meta.x_feats+meta.y_feats
        return df, meta

    @staticmethod
    def _load_mini(meta: MetaData) -> TYPE_DATA_AND_META:
        df = pd.read_csv(meta.url, header=None, sep=' ', skipinitialspace=True, skiprows=[0])
        df['event_type'] = ['signal'] * 36499 + ['background'] * 93565  # Numbers are given at the first line of file
        df = df.rename(columns={i: f for i, f in enumerate(meta.x_feats)})
        return df, meta

    @staticmethod
    def _load_bankruptcy(meta: MetaData, tmp_dname: str = None) -> TYPE_DATA_AND_META:
        import arff

        data_name = 'bankruptcy'
        tmp_dname = f'tmp_{data_name}' if not tmp_dname else tmp_dname

        assert not os.path.isdir(tmp_dname), f'Directory {tmp_dname} already exists.' \
                                             f' Remove this directory or specify another one via `tmp_dname` parameter'

        data = request.urlopen(meta.url).read()
        zf = ZipFile(BytesIO(data))
        zf.extractall(tmp_dname)
        dfs = []
        for fname in os.listdir(tmp_dname):
            data = arff.load(open(f'{tmp_dname}/' + fname, 'r'))
            os.remove(f'{tmp_dname}/' + fname)

            df = pd.DataFrame(data['data'])
            dfs.append(df)
        os.rmdir(tmp_dname)
        df = pd.concat(dfs).reset_index(drop=True)
        del dfs

        meta.x_feats = [f"X{i+1}" for i in range(df.shape[1]-1)]
    
        df.columns = meta.x_feats + meta.y_feats
        return df, meta

    @staticmethod
    def _load_gas_sensor(meta: MetaData, tmp_dname: str = None) -> TYPE_DATA_AND_META:
        tmp_dname = f'tmp_{meta.name}' if not tmp_dname else tmp_dname
        assert not os.path.isdir(tmp_dname), f'Directory {tmp_dname} already exists.' \
                                             f' Remove this directory or specify another one via `tmp_dname` parameter'

        data = request.urlopen(meta.url).read()
        zf = ZipFile(BytesIO(data))
        zf.extractall(tmp_dname)
        dfs = []
        for fname in os.listdir(tmp_dname):
            df = pd.read_csv(f"{tmp_dname}/{fname}", sep=' ', header=None)
            df['batch'] = int(fname.replace('batch', '').replace('.dat', ''))
            os.remove(f"{tmp_dname}/{fname}")
            dfs.append(df)
        df = pd.concat(dfs).reset_index(drop=True)
        del dfs
        os.rmdir(tmp_dname)

        df['gas'] = [meta.classes[int(x.split(';')[0]) - 1] for x in df[0]]
        df['gas_concentration'] = [float(x.split(';')[1]) for x in df[0]]
        df = df.drop([0, 129], axis=1)

        df = df.rename(columns={i+1: f for i, f in enumerate(meta.x_feats)})

        for f in meta.x_feats:
            df[f] = [float(x.split(':')[1]) for x in df[f]]

        return df, meta

    @staticmethod
    def _load_avila(meta: MetaData, tmp_dname: str = None) -> TYPE_DATA_AND_META:
        tmp_dname = f"temp_{meta.name}" if tmp_dname is None else tmp_dname
        assert not os.path.isdir(tmp_dname), f'Directory {tmp_dname} already exists.' \
                                             f' Remove this directory or specify another one via `tmp_dname` parameter'

        data = request.urlopen(meta.url).read()
        zf = ZipFile(BytesIO(data))
        zf.extractall(tmp_dname)

        dfs = []
        for part, part_shrt in [('train', 'tr'), ('test', 'ts')]:
            fname = f"{tmp_dname}/avila/avila-{part_shrt}.txt"
            df = pd.read_csv(fname, header=None)
            os.remove(fname)
            df['dataset_part'] = part
            dfs.append(df)
        df = pd.concat(dfs).reset_index(drop=True)
        del dfs
        os.remove(f"{tmp_dname}/avila/avila-description.txt")
        os.rmdir(f"{tmp_dname}/avila")
        os.rmdir(tmp_dname)

        df = df.rename(columns={i: f for i, f in enumerate(meta.x_feats + meta.y_feats)})
        return df, meta

    @staticmethod
    def _load_workloads(meta: MetaData, tmp_dname: str = None) -> TYPE_DATA_AND_META:
        tmp_dname = f"temp_{meta.name}" if tmp_dname is None else tmp_dname
        assert not os.path.isdir(tmp_dname), f'Directory {tmp_dname} already exists.' \
                                             f' Remove this directory or specify another one via `tmp_dname` parameter'

        data = request.urlopen(meta.url).read()
        zf = ZipFile(BytesIO(data))
        zf.extractall(tmp_dname)

        df = pd.read_csv(f'{tmp_dname}/Datasets/Range-Queries-Aggregates.csv')
        for fname in os.listdir(f"{tmp_dname}/Datasets"):
            os.remove(f"{tmp_dname}/Datasets/{fname}")
        os.rmdir(f"{tmp_dname}/Datasets")
        os.rmdir(f'{tmp_dname}')

        df = df.rename(columns={'Unnamed: 0': 'ID'}).set_index('ID')
        return df, meta

    @staticmethod
    def _load_waveform(meta: MetaData) -> TYPE_DATA_AND_META:
        import unlzw3

        data = request.urlopen(meta.url).read()
        uncompressed_data = unlzw3.unlzw(data)
        df = pd.read_csv(StringIO(uncompressed_data.decode()), header=None)

        df.columns = meta.x_feats+meta.y_feats
        return df, meta

    @staticmethod
    def _load_shuttle(meta: MetaData) -> TYPE_DATA_AND_META:
        import unlzw3

        url_template = meta.url

        url_train = url_template.format(part_novowel='trn.Z')
        data = request.urlopen(url_train).read()
        uncompressed_data = unlzw3.unlzw(data)
        df_train = pd.read_csv(StringIO(uncompressed_data.decode()), header=None, sep=' ')
        df_train['dataset_part'] = 'train'

        url_test = url_template.format(part_novowel='tst')
        df_test = pd.read_csv(url_test, header=None, sep=' ')
        df_test['dataset_part'] = 'test'

        df = pd.concat([df_train, df_test]).reset_index(drop=True)

        df[9] = [meta.classes[i - 1] for i in df[9]]

        df = df.rename(columns={i: f for i, f in enumerate(['time'] + meta.x_feats + meta.y_feats)})
        return df, meta
