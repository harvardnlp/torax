from setuptools import setup

setup(
    name="torax",
    version="0.1",
    packages=["torax"],
    package_data={"torax": []},
    url="https://github.com/harvardnlp/torax",
    install_requires=["jax", "jaxlib"],
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],
)
