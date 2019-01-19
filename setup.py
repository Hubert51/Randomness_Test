from distutils.core import setup
from Cython.Build import cythonize

ext_options = {"compiler_directives": {"profile": True}, "annotate": True}
setup(
    ext_modules = cythonize("python_code/PyRandomUtils.pyx", **ext_options)
)

