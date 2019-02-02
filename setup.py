from distutils.core import setup
from Cython.Build import cythonize
import numpy
# python setup.py build_ext --inplace
ext_options = {"compiler_directives": {"profile": True, "language_level" : "3"}, "annotate": True}
setup(
    ext_modules = cythonize("python_code/PyRandomUtils.pyx", **ext_options),
    include_dirs=[numpy.get_include()]
)

