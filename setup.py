from setuptools import setup,find_packages


setup(
	name='main',
	version='1.0',
	description='a python package for translation between two languages',
	author='Amine AMMOR',
	packages=find_packages(where='src'),
    package_dir = {'':'src'}
)
