from distutils.core import setup
from distutils.extension import  Extension
from Cython.Build import cythonize
import numpy
# python setup.py build_ext --inplace
ext_options = {"compiler_directives": {"profile": True, "language_level" : "3"}, "annotate": True}
ext_modules =  [Extension('utils.PyRandomUtils',
                          ['utils/PyRandomUtils.pyx']),
                Extension('utils.CardUtils',
                          ['utils/CardUtils.pyx'])]

setup(
    ext_modules = cythonize(ext_modules, **ext_options),
    include_dirs=[numpy.get_include()]
)

