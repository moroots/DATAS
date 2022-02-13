# -*- coding: utf-8 -*-
"""
Created on Sat Feb 12 15:15:42 2022

@author: Maurice Roots

"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open(r"datas/requirements.txt", "r") as f:
    reqs = f.read()

reqs = reqs.replace(" ", "")
reqs = reqs.split("\n")
reqs = [ x for x in reqs if "#" not in x ]
reqs = ' '.join(reqs)
reqs = reqs.split()

setup(
    name='datas',
    version='0.0.1',
    author='Maurice Roots',
    author_email='mroots1@umbc.edu',
    description='Data Analysis Tools for Atmsopheric Science',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/moroots/DATAS',
    project_urls = {
        "Bug Tracker": "https://github.com/moroots/DATAS/issues"
    },
    license='MIT',
    packages=find_packages(),
    install_requires=reqs,
)



