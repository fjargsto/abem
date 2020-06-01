from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [Extension("*",  sources=["src/*.pyx"], include_dirs=[numpy.get_include()],
                        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]),
              Extension("iops_cpp", sources=["iops_cpp/helmholtz_integrals.pyx", "iops_cpp/iops_cpp.cpp"],
                        language="c++", )]


def readme():
    with open("README.md", "r") as fh:
        return fh.read()


def requirements():
    # The dependencies are the same as the contents of requirements.txt
    with open("requirements.txt") as f:
        return [line.strip() for line in f if line.strip()]


setup(
    name="abem",
    version="0.2b2",
    description="Boundary Element Method for Acoustic Simulations",
    long_description=readme(),
    long_description_content_type="text/markdown",
    url="http://github.com/fjargsto/abem",
    author="Frank Jargstorff",
    download_url="https://github.com/fjargsto/abem/archive/abem-0.2a2.tar.gz",
    license="GNU General Public License",
    packages=["abem", "iops_sci"],
    install_requires=requirements(),
    zip_safe=False,
    ext_modules=cythonize(extensions),
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],
    test_suite="tests",
    data_files=[
        (
            "notebooks",
            [
                "notebooks/exterior_helmholtz_solver_2d.ipynb",
                "notebooks/exterior_helmholtz_solver_3d.ipynb",
                "notebooks/exterior_helmholtz_solver_rad.ipynb",
                "notebooks/interior_helmholtz_solver_2d.ipynb",
                "notebooks/interior_helmholtz_solver_3d.ipynb",
                "notebooks/interior_helmholtz_solver_rad.ipynb",
                "notebooks/rayleigh_cavity_1.ipynb",
                "notebooks/rayleigh_cavity_2.ipynb",
                "notebooks/rayleigh_solver_3d_disk.ipynb",
                "notebooks/rayleigh_solver_square.ipynb",
            ],
        )
    ],
)
