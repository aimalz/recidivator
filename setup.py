from setuptools import setup

setup(
    name="recidivator",
    version="0.1",
    author="Alex Malz",
    author_email="aimalz@astro.rub.de",
    url = "https://github.com/aimalz/recidivator",
    packages=["recidivator"],
    description="HOD without halos",
    long_description=open("README.md").read(),
    package_data={"": ["README.md", "LICENSE"]},
    include_package_data=True,
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        ],
    install_requires=["matplotlib", "numpy"]
)
