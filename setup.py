from setuptools import setup, Extension

intops = Extension('intops',
                   sources=['intops/intops.c'],
                   language='c', )

def readme():
    with open("README.md", "r") as fh:
        return fh.read()

def requirements():
    # The dependencies are the same as the contents of requirements.txt
    with open('requirements.txt') as f:
        return [line.strip() for line in f if line.strip()]


setup(
    name='abem',
    version='0.1-alpha',
    description='Boundary Element Method for Acoustic Simulations',
    long_description=readme(),
    long_description_content_type='text/markdown',
    url='http://github.com/fjargsto/abem',
    author='Frank Jargstorff',
    download_url='https://github.com/fjargsto/abem/archive/v0.1-alpha.tar.gz',
    license='GNU General Public License',
    packages=['abem'],
    install_requires=requirements(),
    zip_safe=False,
    ext_modules=[intops],
    test_suite='test',
    data_files=[(
        'notebooks', [
            'notebooks/exterior_helmholtz_solver_2d.ipynb',
            'notebooks/exterior_helmholtz_solver_3d.ipynb',
            'notebooks/exterior_helmholtz_solver_rad.ipynb',
            'notebooks/interior_helmholtz_solver_2d.ipynb',
            'notebooks/interior_helmholtz_solver_3d.ipynb',
            'notebooks/interior_helmholtz_solver_rad.ipynb',
            'notebooks/rayleigh_cavity_1.ipynb',
            'notebooks/rayleigh_cavity_2.ipynb',
            'notebooks/rayleigh_solver_3d_disk.ipynb',
            'notebooks/rayleigh_solver_square.ipynb']
    )],
)
