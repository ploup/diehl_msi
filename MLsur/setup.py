from setuptools import setup

setup(
    name='module_diehl_msi',
    version='0.1.0',
    description="Scripts python développé en tant qu'IE au labo Duval",
    author='Aurelien DIEHL',
    author_email='aurelien.diehl@inserm.fr',
    url='https://github.com/ploup/MSI_python_scripts.git',
    packages=["MLsur","resfile","RNAseq"],
    install_requires=['pandas','numpy'],
    # entry_points={
    #     'console_scripts': ['mon_module = mon_module.main'],
    # },
)