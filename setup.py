from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "sublinear_cpp._core",
        ["src/sublinear_cpp/_core.cpp"],
        cxx_std=14,
        extra_compile_args=["-O3"],
    ),
]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
