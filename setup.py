import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="yellowbrick_plugins",
    version="0.0.1",
    author="Hung-Tien Huang",
    author_email="hungtienhuang@gmail.com",
    description=
    "yellowbrick_plugins extends yellowbrick package to provide model selection for modules defined in sklearn_plugins package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/busFred/yellowbrick_plugins",
    project_urls={
        "Bug Tracker": "https://github.com/busFred/yellowbrick_plugins/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
    install_requires=[
        'numpy', 'scipy', 'scikit-learn', 'skl2onnx', 'sklearn_plugins', 'yellowbrick'
    ])
