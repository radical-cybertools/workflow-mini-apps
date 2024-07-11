from setuptools import setup, find_packages

setup(
    name='wfminiAPI',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'cupy',
        'h5py',
        'mpi4py'
    ],
    author='Tianle Wang, Ozgur Kilic',
    author_email='twang3@bnl.gov',
    description='An open source library that is used to make implementing emulated task in workflow mini-app simple. It support both Python and C++ (OpenMP) backend and is targetting various different architecture, including CPU, NVIDIA GPU, AMD GPU and Intel GPU',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    license='MIT',
    keywords='example keywords',
    url='',
)
