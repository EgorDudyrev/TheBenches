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
    x_feats: List[str]
    y_feats: Optional[List[str]] = None


TYPE_DATA_AND_META = Tuple[pd.DataFrame, MetaData]


class DataHandler:
    def __init__(self):
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
            iris='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
            wine='https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data',
            haberman='https://archive.ics.uci.edu/ml/machine-learning-databases/haberman/haberman.data',
            ecoli='https://archive.ics.uci.edu/ml/machine-learning-databases/ecoli/ecoli.data',
            breast_w='https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data',
            spambase='https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data',
            waveform='https://archive.ics.uci.edu/ml/machine-learning-databases/waveform/waveform.data.Z',
            parkinsons='https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data',
            statlog='https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/satimage/sat.{part_novowel}',
            gas_sensor='https://archive.ics.uci.edu/ml/machine-learning-databases/00270/driftdataset.zip',
            avila="https://archive.ics.uci.edu/ml/machine-learning-databases/00459/avila.zip",
            credit_card='https://archive.ics.uci.edu/ml/machine-learning-databases/00350/'
                        'default%20of%20credit%20card%20clients.xls',
            shuttle='https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/shuttle/shuttle.{part_novowel}',
            sensorless='https://archive.ics.uci.edu/ml/machine-learning-databases/00325/Sensorless_drive_diagnosis.txt',
            mini='https://archive.ics.uci.edu/ml/machine-learning-databases/00199/MiniBooNE_PID.txt',
            workloads='https://archive.ics.uci.edu/ml/machine-learning-databases/00493/datasets.zip',
        )

    @property
    def available_datasets(self):
        return tuple(self._raw_url_selector.keys())

    def load_dataset(self, data_name: str) -> TYPE_DATA_AND_META:
        df, meta = self.func_selector(data_name)()

        assert len(set(meta.x_feats)) == len(meta.x_feats)
        if meta.y_feats:
            assert len(set(meta.y_feats)) == len(meta.y_feats)

        return df, meta

    @staticmethod
    def load_metadata(data_name: str) -> MetaData:
        stream = pkg_resources.resource_stream(__name__, f'meta_data/{data_name}.json')
        dct = json.loads(stream.read())
        dct['name'] = data_name
        return MetaData.from_dict(dct)

    def func_selector(self, data_name: str) -> Callable[[], TYPE_DATA_AND_META]:
        if data_name in {'animal_movement', 'mango'}:
            return lambda: self._load_csv_from_fcapy(data_name)
        if data_name in {'live_in_water', 'digits', 'gewaesser', 'lattice', 'tealady'}:
            return lambda: self._load_cxt_from_fcapy(data_name)

        return self.__getattribute__(f"_load_{data_name}")

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
            'zab_leg_04', 'zab_leg_06', 'S_AD_KBRIG', 'D_AD_KBRIG', 'S_AD_ORIT', 'D_AD_ORIT', 'O_L_POST', 'K_SH_POST',
            'MP_TP_POST', 'SVT_POST', 'GT_POST', 'FIB_G_POST', 'ant_im', 'lat_im', 'inf_im', 'post_im', 'IM_PG_P',
            'ritm_ecg_p_01', 'ritm_ecg_p_02', 'ritm_ecg_p_04', 'ritm_ecg_p_06', 'ritm_ecg_p_07', 'ritm_ecg_p_08',
            'n_r_ecg_p_01', 'n_r_ecg_p_02', 'n_r_ecg_p_03', 'n_r_ecg_p_04', 'n_r_ecg_p_05', 'n_r_ecg_p_06',
            'n_r_ecg_p_08', 'n_r_ecg_p_09', 'n_r_ecg_p_10', 'n_p_ecg_p_01', 'n_p_ecg_p_03', 'n_p_ecg_p_04',
            'n_p_ecg_p_05', 'n_p_ecg_p_06', 'n_p_ecg_p_07', 'n_p_ecg_p_08', 'n_p_ecg_p_09', 'n_p_ecg_p_10',
            'n_p_ecg_p_11', 'n_p_ecg_p_12', 'fibr_ter_01', 'fibr_ter_02', 'fibr_ter_03', 'fibr_ter_05', 'fibr_ter_06',
            'fibr_ter_07', 'fibr_ter_08', 'GIPO_K', 'K_BLOOD', 'GIPER_Na', 'Na_BLOOD', 'ALT_BLOOD', 'AST_BLOOD',
            'KFK_BLOOD', 'L_BLOOD', 'ROE', 'TIME_B_S', 'R_AB_1_n', 'R_AB_2_n', 'R_AB_3_n', 'NA_KB', 'NOT_NA_KB',
            'LID_KB', 'NITR_S', 'NA_R_1_n', 'NA_R_2_n', 'NA_R_3_n', 'NOT_NA_1_n', 'NOT_NA_2_n', 'NOT_NA_3_n', 'LID_S_n',
            'B_BLOK_S_n', 'ANT_CA_S_n', 'GEPAR_S_n', 'ASP_S_n', 'TIKL_S_n', 'TRENT_S_n'
        ]
        y_feats = [
            'FIBR_PREDS', 'PREDS_TAH', 'JELUD_TAH', 'FIBR_JELUD', 'A_V_BLOK', 'OTEK_LANC', 'RAZRIV', 'DRESSLER',
            'ZSN', 'REC_IM', 'P_IM_STEN', 'LET_IS'
        ]
        df.index.name = 'ID'
        df.columns = x_feats+y_feats
        meta = MetaData(name=data_name, url=url, x_feats=x_feats, y_feats=y_feats)
        return df, meta

    def _load_iris(self) -> TYPE_DATA_AND_META:
        data_name = 'iris'

        url = self._raw_url_selector[data_name]
        df = pd.read_csv(url, header=None)
        x_feats = ['sepal length in cm', 'sepal width in cm', 'petal length in cm', 'petal width in cm']
        y_feats = ['class']
        df.index.name = 'ID'
        df.columns = x_feats+y_feats
        meta = MetaData(name=data_name, url=url, x_feats=x_feats, y_feats=y_feats)
        return df, meta

    def _load_wine(self) -> TYPE_DATA_AND_META:
        data_name = 'wine'

        url = self._raw_url_selector[data_name]
        df = pd.read_csv(url, header=None)
        x_feats = [
            'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols',
            'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue',
            'OD280/OD315 of diluted wines', 'Proline'
        ]
        y_feats = ['class']
        df.index.name = 'ID'
        df.columns = x_feats + y_feats
        meta = MetaData(name=data_name, url=url, x_feats=x_feats, y_feats=y_feats)
        return df, meta

    def _load_haberman(self) -> TYPE_DATA_AND_META:
        data_name = 'haberman'

        url = self._raw_url_selector[data_name]
        df = pd.read_csv(url, header=None)
        x_feats = [
            'Age of patient at time of operation', "Patient's year of operation",
            "Number of positive axillary nodes detected",
        ]
        y_feats = ['Survival status']
        df.index.name = 'ID'
        df.columns = x_feats + y_feats
        meta = MetaData(name=data_name, url=url, x_feats=x_feats, y_feats=y_feats)
        return df, meta

    def _load_ecoli(self) -> TYPE_DATA_AND_META:
        data_name = 'ecoli'

        url = self._raw_url_selector[data_name]
        response = request.urlopen(url)
        s = re.sub(r"\ +", "\t", response.read().decode())
        df = pd.read_table(StringIO(s), sep='\t', header=None)
        x_feats = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2']
        y_feats = ['Class']
        df.index.name = 'ID'
        df.columns = x_feats + y_feats
        meta = MetaData(name=data_name, url=url, x_feats=x_feats, y_feats=y_feats)
        return df, meta

    def _load_breast_w(self) -> TYPE_DATA_AND_META:
        data_name = 'breast_w'

        url = self._raw_url_selector[data_name]
        x_feats_base = [
            'radius (mean of distances from center to points on the perimeter)',
            'texture (standard deviation of gray-scale values)',
            'perimeter', 'area',
            'smoothness (local variation in radius lengths)',
            'compactness (perimeter^2 / area - 1.0)',
            'concavity (severity of concave portions of the contour)',
            'concave points (number of concave portions of the contour)',
            'symmetry',
            'fractal dimension ("coastline approximation" - 1)'
        ]
        x_feats = [f"{t} {f}" for t in ['Mean', 'SError', 'Worst'] for f in x_feats_base]
        y_feats = ['Diagnosis']

        df = pd.read_csv(url, header=None)
        df.columns = ['ID Number']+y_feats+x_feats
        df = df.set_index('ID Number')
        meta = MetaData(name=data_name, url=url, x_feats=x_feats, y_feats=y_feats)
        return df, meta

    def _load_spambase(self) -> TYPE_DATA_AND_META:
        data_name = 'spambase'

        x_feats = [
            'word_freq_make', 'word_freq_address', 'word_freq_all', 'word_freq_3d', 'word_freq_our', 'word_freq_over',
            'word_freq_remove', 'word_freq_internet', 'word_freq_order', 'word_freq_mail', 'word_freq_receive',
            'word_freq_will', 'word_freq_people', 'word_freq_report', 'word_freq_addresses', 'word_freq_free',
            'word_freq_business', 'word_freq_email', 'word_freq_you', 'word_freq_credit', 'word_freq_your',
            'word_freq_font', 'word_freq_000', 'word_freq_money', 'word_freq_hp', 'word_freq_hpl', 'word_freq_george',
            'word_freq_650', 'word_freq_lab', 'word_freq_labs', 'word_freq_telnet', 'word_freq_857', 'word_freq_data',
            'word_freq_415', 'word_freq_85', 'word_freq_technology', 'word_freq_1999', 'word_freq_parts', 'word_freq_pm',
            'word_freq_direct', 'word_freq_cs', 'word_freq_meeting', 'word_freq_original', 'word_freq_project',
            'word_freq_re', 'word_freq_edu', 'word_freq_table', 'word_freq_conference', 'char_freq_;', 'char_freq_(',
            'char_freq_[', 'char_freq_!', 'char_freq_$', 'char_freq_#', 'capital_run_length_average',
            'capital_run_length_longest', 'capital_run_length_total'
        ]
        y_feats = ['spam']

        url = self._raw_url_selector[data_name]
        df = pd.read_csv(url, header=None)
        df.columns = x_feats + y_feats
        meta = MetaData(name=data_name, url=url, x_feats=x_feats, y_feats=y_feats)
        return df, meta

    def _load_parkinsons(self) -> TYPE_DATA_AND_META:
        data_name = 'parkinsons'

        url = self._raw_url_selector[data_name]
        df = pd.read_csv(url)
        df = df.set_index('name')

        y_feats = ['status']
        x_feats = list(df.drop(y_feats, axis=1).columns)
        meta = MetaData(name=data_name, url=url, x_feats=x_feats, y_feats=y_feats)
        return df, meta

    def _load_statlog(self) -> TYPE_DATA_AND_META:
        data_name = 'statlog'

        pixels = [f"{v}-{h}" for v in ['top', 'middle', 'bottom'] for h in ['left', 'middle', 'right']]
        x_feats = [f"spectrum_{sp}_{px}" for sp in range(1, 5) for px in pixels]
        y_feats = ['class']
        classes = ['red soil', 'cotton crop', 'grey soil', 'damp grey soil', 'soil with vegetation stubble',
                   'mixture class (all types present)', 'very damp grey soil']

        dfs = []
        for part, part_novowel in [('train', 'trn'), ('test', 'tst')]:
            url = self._raw_url_selector[data_name].format(part_novowel=part_novowel)
            df = pd.read_csv(url, sep=' ', header=None)
            df.columns = x_feats + y_feats
            df['class'] = [classes[i - 1] for i in df['class']]
            df['dataset_part'] = part
            dfs.append(df)
        df = pd.concat(dfs).reset_index(drop=True)

        meta = MetaData(name=data_name, url=url, x_feats=x_feats, y_feats=y_feats)
        return df, meta

    def _load_credit_card(self) -> TYPE_DATA_AND_META:
        data_name = 'credit_card'

        url = self._raw_url_selector[data_name]
        df = pd.read_excel(url, header=1).set_index('ID')
        y_feats = ['default payment next month']
        x_feats = list(df.drop(y_feats, axis=1).columns)

        meta = MetaData(name=data_name, url=url, x_feats=x_feats, y_feats=y_feats)
        return df, meta

    def _load_sensorless(self) -> TYPE_DATA_AND_META:
        data_name = 'sensorless'

        url = self._raw_url_selector[data_name]
        df = pd.read_csv(url, header=None, sep=' ')

        # Some combination of "The first three intrinsic mode functions (IMF)
        # of the two phase currents and their residuals (RES) were used and broken down into sub-sequences"
        base_feats = [f"X{i}" for i in range(1, 13)]
        x_feats = [f"{op}_{bf}" for op in ['mean', 'std', 'skewness', 'kurtosis'] for bf in base_feats]
        y_feats = ['class']
        df.columns = x_feats+y_feats
        meta = MetaData(name=data_name, url=url, x_feats=x_feats, y_feats=y_feats)
        return df, meta

    def _load_mini(self) -> TYPE_DATA_AND_META:
        data_name = 'mini'

        url = self._raw_url_selector[data_name]
        df = pd.read_csv(url, header=None, sep=' ', skipinitialspace=True, skiprows=[0])
        df['event_type'] = ['signal'] * 36499 + ['background'] * 93565  # Numbers are given at the first line of file

        x_feats = [f"Particle ID {i}" for i in range(1, 51)]
        y_feats = ['event_type']
        df = df.rename(columns={i: f for i, f in enumerate(x_feats)})

        meta = MetaData(name=data_name, url=url, x_feats=x_feats, y_feats=y_feats)
        return df, meta

    def _load_bankruptcy(self, tmp_dname: str = None) -> TYPE_DATA_AND_META:
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
        df = pd.concat(dfs).reset_index(drop=True)
        del dfs

        x_feats = [f"X{i+1}" for i in range(df.shape[1]-1)]
        y_feats = ['is_bankrupted']
        df.columns = x_feats+y_feats
        meta = MetaData(name=data_name, url=url, x_feats=x_feats, y_feats=y_feats)

        return df, meta

    def _load_gas_sensor(self, tmp_dname: str = None) -> TYPE_DATA_AND_META:
        data_name = 'gas_sensor'
        tmp_dname = f'tmp_{data_name}' if not tmp_dname else tmp_dname
        assert not os.path.isdir(tmp_dname), f'Directory {tmp_dname} already exists.' \
                                             f' Remove this directory or specify another one via `tmp_dname` parameter'

        base_feats = ['DR_{i}', '|DR|_{i}', 'EMAi0.001_{i}', 'EMAi0.01_{i}', 'EMAi0.1_{i}', 'EMAd0.001_{i}',
                      'EMAd0.01_{i}', 'EMAd0.1_{i}']

        url = self._raw_url_selector[data_name]
        data = request.urlopen(url).read()
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

        classes = ['Ethanol', 'Ethylene', 'Ammonia', 'Acetaldehyde', 'Acetone', 'Toluene']
        df['gas'] = [classes[int(x.split(';')[0]) - 1] for x in df[0]]
        df['gas_concentration'] = [float(x.split(';')[1]) for x in df[0]]
        df = df.drop([0, 129], axis=1)

        x_feats = [f.format(i=i) for i in range(1, 17) for f in base_feats]
        y_feats = ['gas']
        df = df.rename(columns={i+1: f for i, f in enumerate(x_feats)})

        for f in x_feats:
            df[f] = [float(x.split(':')[1]) for x in df[f]]

        meta = MetaData(name=data_name, url=url, x_feats=x_feats, y_feats=y_feats)
        return df, meta

    def _load_avila(self, tmp_dname: str = None) -> TYPE_DATA_AND_META:
        data_name = 'avila'
        tmp_dname = f"temp_{data_name}" if tmp_dname is None else tmp_dname

        url = self._raw_url_selector[data_name]
        data = request.urlopen(url).read()
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

        x_feats = ['F1: intercolumnar distance', 'F2: upper margin', 'F3: lower margin', 'F4: exploitation',
                   'F5: row number', 'F6: modular ratio', 'F7: interlinear spacing', 'F8: weight', 'F9: peak number',
                   'F10: modular ratio/ interlinear spacing']
        y_feats = ['Class']
        df = df.rename(columns={i: f for i, f in enumerate(x_feats + y_feats)})

        meta = MetaData(name=data_name, url=url, x_feats=x_feats, y_feats=y_feats)
        return df, meta

    def _load_workloads(self, tmp_dname: str = None) -> TYPE_DATA_AND_META:
        data_name = 'workloads'
        tmp_dname = f"temp_{data_name}" if tmp_dname is None else tmp_dname

        url = self._raw_url_selector[data_name]
        data = request.urlopen(url).read()
        zf = ZipFile(BytesIO(data))
        zf.extractall(tmp_dname)

        df = pd.read_csv(f'{tmp_dname}/Datasets/Range-Queries-Aggregates.csv')
        for fname in os.listdir(f"{tmp_dname}/Datasets"):
            os.remove(f"{tmp_dname}/Datasets/{fname}")
        os.rmdir(f"{tmp_dname}/Datasets")
        os.rmdir(f'{tmp_dname}')

        df = df.rename(columns={'Unnamed: 0': 'ID'}).set_index('ID')

        x_feats = ['x', 'y', 'x_range', 'y_range', 'count', 'sum_', 'avg']
        y_feats = None  # Maybe ['count', 'sum_', 'avg'] ?

        meta = MetaData(name=data_name, url=url, x_feats=x_feats, y_feats=y_feats)
        return df, meta

    def _load_waveform(self) -> TYPE_DATA_AND_META:
        import unlzw3

        data_name = 'waveform'

        url = self._raw_url_selector[data_name]
        data = request.urlopen(url).read()
        uncompressed_data = unlzw3.unlzw(data)
        df = pd.read_csv(StringIO(uncompressed_data.decode()), header=None)

        x_feats = [f"f{i}" for i in range(21)]
        y_feats = ['class']
        df.columns = x_feats+y_feats
        df.index.name = 'ID'
        meta = MetaData(name=data_name, url=url, x_feats=x_feats, y_feats=y_feats)
        return df, meta

    def _load_shuttle(self) -> TYPE_DATA_AND_META:
        import unlzw3

        data_name = 'shuttle'
        url_template = self._raw_url_selector[data_name]

        url_train = url_template.format(part_novowel='trn.Z')
        data = request.urlopen(url_train).read()
        uncompressed_data = unlzw3.unlzw(data)
        df_train = pd.read_csv(StringIO(uncompressed_data.decode()), header=None, sep=' ')
        df_train['dataset_part'] = 'train'

        url_test = url_template.format(part_novowel='tst')
        df_test = pd.read_csv(url_test, header=None, sep=' ')
        df_test['dataset_part'] = 'test'

        df = pd.concat([df_train, df_test]).reset_index(drop=True)

        classes = ['Rad Flow', 'Fpv Close', 'Fpv Open', 'High', 'Bypass', 'Bpv Close', 'Bpv Open']
        df[9] = [classes[i - 1] for i in df[9]]

        x_feats = [f'X{i}' for i in range(1, 9)]
        y_feats = ['class']
        df = df.rename(columns={i: f for i, f in enumerate(['time'] + x_feats + y_feats)})

        meta = MetaData(name=data_name, url=url_template, x_feats=x_feats, y_feats=y_feats)
        return df, meta
