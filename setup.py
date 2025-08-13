from setuptools import setup, find_packages

setup(
    name="cad2sees",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "openseespy",
        "pyvista",
        "matplotlib",
        "scipy"
    ]
)