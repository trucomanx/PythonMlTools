#!/usr/bin/python

from setuptools import setup, find_packages

setup(
    name   ='PythonMlTools',
    version='0.1.1',
    author='Fernando Pujaico Rivera',
    author_email='fernando.pujaico.rivera@gmail.com',
    packages=[  'PythonMlTools',
                'PythonMlTools.Regression',
                'PythonMlTools.Classification',
                'PythonMlTools.DataSet',
                'PythonMlTools.DensityEstimation',
                'PythonMlTools.FeatureSelection',
                'PythonMlTools.FeatureSelection.Analysis',
                'PythonMlTools.FeatureSelection.Metrics'],
    #scripts=['bin/script1','bin/script2'],
    url='https://github.com/trucomanx/PythonMlTools',
    license='GPLv3',
    description='Machine learning tools',
    #long_description=open('README.txt').read(),
    install_requires=[
       "scikit-learn", #"Django >= 1.1.1",
       "tqdm", #
       "numpy"
    ],
)

#! python setup.py sdist bdist_wheel
# Upload to PyPi
# or 
#! pip3 install dist/PythonMlTools-0.1.tar.gz 
