import setuptools


def run_install(**kwargs):
    with open("README.md", "r") as fh:
        long_description = fh.read()

    extras_require = {
        'datahandler': [
            'pandas',
            'liac-arff',
            'unlzw3',
            'xlrd',
        ],
        # 'docs': [
        #     'numpydoc',
        #     'sphinx_rtd_theme',
        #     'sphinx',
        #     'nbsphinx',
        # ]
    }
    extras_require['all'] = list(set(i for val in extras_require.values() for i in val))
    # extras_require['docs'] = extras_require['all']

    setuptools.setup(
        name="benches",
        version="0.1.0",
        author="Egor Dudyrev",
        author_email="egor.dudyrev@yandex.ru",
        description="A python package to make data science experiments high and dry",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/EgorDudyrev/TheBenches",
        packages=setuptools.find_packages(exclude=("tests",)),
        classifiers=[
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.8",
            "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
            "Operating System :: OS Independent",
        ],
        python_requires='>=3.8',
        extras_require=extras_require
    )


if __name__ == "__main__":
    run_install()
