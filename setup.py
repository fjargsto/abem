from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy


intops = Extension("iops_cpp",
                   sources=[
                       "iops_cpp/helmholtz_integrals.pyx",
                       "iops_cpp/iops_cpp.cpp"],
                   language="c++",)

pyntops = Extension("iops_pyx",
                    sources=["iops_pyx/helmholtz_integrals.pyx"],
                    include_dirs=[numpy.get_include()])

#scintops = Extension("iops_sci",
#                    sources=["iops_sci/helmholtz_integrals.pyx"],
#                    include_dirs=[numpy.get_include()])

aprops = Extension("aprops",
                    sources=["aprops/acoustic_properties.pyx"],
                    include_dirs=[numpy.get_include()])

abounds = Extension("abounds",
                    sources=["abounds/mesh.pyx"],
                    include_dirs=[numpy.get_include()])

solver = Extension("solver",
                    sources=["solver/solver.pyx"],
                    include_dirs=[numpy.get_include()])

hsolvers = Extension("hsolvers",
                    sources=["hsolvers/helmholtz_solver.pyx"],
                    include_dirs=[numpy.get_include()])

rsolvers = Extension("rsolvers",
                    sources=["rsolvers/rayleigh_solvers.pyx"],
                    include_dirs=[numpy.get_include()])


def readme():
    with open("README.md", "r") as fh:
        return fh.read()


def requirements():
    # The dependencies are the same as the contents of requirements.txt
    with open("requirements.txt") as f:
        return [line.strip() for line in f if line.strip()]


setup(
    name="abem",
    version="0.1-alpha",
    description="Boundary Element Method for Acoustic Simulations",
    long_description=readme(),
    long_description_content_type="text/markdown",
    url="http://github.com/fjargsto/abem",
    author="Frank Jargstorff",
    download_url="https://github.com/fjargsto/abem/archive/v0.1-alpha.tar.gz",
    license="GNU General Public License",
    packages=["abem"],
    install_requires=requirements(),
    zip_safe=False,
    ext_modules=cythonize([intops, pyntops, aprops, abounds, solver, hsolvers, rsolvers],
                          include_path=["intops", "pyntops"]),
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
