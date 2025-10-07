from setuptools import setup, find_packages

setup(
    name="d3il_lite.d3il_sim",
    version="0.2",
    description="Franka Panda Simulators",
    license="MIT",
    package_data={"models": ["*"]},
    packages=find_packages(),
)
