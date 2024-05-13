#!/usr/bin/env python3
import os
import site
import sys
from distutils.core import setup

import setuptools

# Editable install in user site directory can be allowed with this hack:
# https://github.com/pypa/pip/issues/7953.
site.ENABLE_USER_SITE = "--user" in sys.argv[1:]

with open("README.md", encoding='utf-8') as f:
    long_description = f.read()

with open(os.path.join("robust_speech", "version.txt"), encoding='utf-8') as f:
    version = f.read().strip()

setup(
    name="robust_speech",
    version=version,
    description="Adversarially Robust Speech Recognition toolkit built over SpeechBrain",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Raphael Olivier",
    author_email="rolivier@cs.cmu.edu",
    packages=setuptools.find_packages(),
    package_data={"robust_speech": ["version.txt", "log-config.yaml"]},
    install_requires=[
        # "torch>=1.8.0,<=1.11",
        # "transformers>=4.20.0",
 #       "speechbrain @ git+https://github.com/RaphaelOlivier/speechbrain.git",
        "audlib @ git+https://github.com/RaphaelOlivier/pyaudlib.git",
    ],
    python_requires=">=3.8",
    url="https://github.com/RaphaelOlivier/robust_speech",
)
