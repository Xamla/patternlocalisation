"""Python package for our asymmetric circle pattern localisation
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))


setup(
    name='patternlocalisation',

    version='0.0.1',

    description='A library for pattern localisation.',
    long_description="A library for detecting our asymmetric circle patterns and calculating the pattern pose in camera coordinates.",

    url='https://github.com/Xamla/patternlocalisation',
    author='Inga Altrogge',
    author_email='inga.altrogge@xamla.com',

    license='see LICENSE file',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Topic :: Pattern Detection',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.1',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6'
    ],
        
        
    # What does your project relate to?
    keywords=[
        'patterndetection', 'patternpose', 'pattern_camera_transformation'
    ],

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=['patternlocalisation'],

    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed.
    install_requires=["numpy"],

    # List additional groups of dependencies here (e.g. development
    # dependencies). You can install these using the following syntax,
    # for example:
    # $ pip install -e .[dev,test]
    extras_require={
    },

    # If there are data files included in your packages that need to be
    # installed, specify them here.  If using Python 2.6 or less, then these
    # have to be included in MANIFEST.in as well.
    #package_dir={'patternlocalisation': 'patternlocalisation'},
    #include_package_data=True,
    package_data={'patternlocalisation': ['*.npy']},

    # Although 'package_data' is the preferred approach, in some case you may
    # need to place data files outside of your packages. See:
    # http://docs.python.org/3.4/distutils/setupscript.html#installing-additional-files # noqa
    # In this case, 'data_file' will be installed into '<sys.prefix>/my_data'
    data_files=[],

    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # pip to create the appropriate form of executable for the target platform.
    entry_points={},

    # Use nose to discover all tests in the module
    test_suite='nose.collector',

    # Set Nose as a requirement for running tests
    tests_require=['nose'],
)
