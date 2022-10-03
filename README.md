# TheBenches
Python package to make data science experiments high and dry

# Installation

Install the package by cloning it from GitHub:

```console
git clone https://github.com/EgorDudyrev/TheBenches.git
pip install TheBenches/
```

There is no pypi version of the package for now.

# Use case

To list all available datasets:

```python
from benches import DataHandler
print(DataHandler.available_datasets())
```
> ('gewaesser', 'mango', 'mini', 'credit_card', 'gas_sensor', 'sensorless', 'waveform', 'statlog', 'live_in_water', 'ecoli', 'avila', 'spambase', 'myocard', 'wine', 'breast_w', 'tealady', 'lattice', 'digits', 'workloads', 'haberman', 'parkinsons', 'iris', 'animal_movement', 'bankruptcy', 'shuttle')

To load one of the datasets:

```python
from benches import DataHandler
dh = DataHandler()                                                      
df, meta = dh.load_dataset('iris')                                      
print(df.head())
print(meta)
```
> <pre>
>    sepal length in cm  sepal width in cm  ...  petal width in cm        class
> 0                 5.1                3.5  ...                0.2  Iris-setosa
> 1                 4.9                3.0  ...                0.2  Iris-setosa
> 2                 4.7                3.2  ...                0.2  Iris-setosa
> 3                 4.6                3.1  ...                0.2  Iris-setosa
> 4                 5.0                3.6  ...                0.2  Iris-setosa
> 
> [5 rows x 5 columns]
> </pre>

> MetaData(name='iris', url='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', loading_function='_load_iris', x_feats=['sepal length in cm', 'sepal width in cm', 'petal length in cm', 'petal width in cm'], y_feats=['class'], classes=None)

