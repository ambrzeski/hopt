import setuptools

setuptools.setup(
    name='hopt',
    version='0.2.1',
    author='Adam Brzeski',
    author_email='brzeski@eti.pg.edu.pl',
    description='Simple framework for hyperparameter optimization and cross-validation',
    long_description='Simple framework for hyperparameter optimization and cross-validation',
    long_description_content_type='text/markdown',
    url='https://github.com/ambrzeski/hopt',
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Natural Language :: English',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Intended Audience :: Developers'
    ],
    install_requires=['keras>=2.0.0']
)
