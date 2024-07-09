# -*- coding: utf-8 -*-
"""
Created on Sat Feb 12 15:15:42 2022

@author: Maurice Roots

"""

from setuptools import setup, find_packages

#%%
 
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open(r"datas/requirements.txt", "r") as f:
    reqs = f.read()

reqs = reqs.replace(" ", "")
reqs = reqs.split("\n")
reqs = [x for x in reqs if "#" not in x ]
reqs = [x for x in reqs if "pyhdf" not in x]
reqs = ' '.join(reqs)
reqs = reqs.split()

setup(
    name='datas',
    version='2024.07.09.01',
    author='Maurice Roots',
    author_email='themauriceroots@gmail.com',
    description='Data Analysis Tools for Atmsopheric Science',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/moroots/DATAS',
    project_urls = {
        "Bug Tracker": "https://github.com/moroots/DATAS/issues"
    },
    license='MIT',
    packages=find_packages(),
    package_data={"datas": ['datas\\sample_data\\__init__.py',
     'datas\\sample_data\\aeronet\\20010101_20241231_CCNY.all',
     'datas\\sample_data\\aeronet\\20010101_20241231_CCNY.zip',
     'datas\\sample_data\\aeronet\\20200101_20241231_GSFC.all',
     'datas\\sample_data\\aeronet\\20200101_20241231_GSFC.zip',
     'datas\\sample_data\\aeronet\\20200101_20241231_Hampton_University.all',
     'datas\\sample_data\\aeronet\\20200101_20241231_Hampton_University.zip',
     'datas\\sample_data\\pandora\\Pandora135s1_ManhattanNY-CCNY_L2_rfus5p1-8.parquet',
     'datas\\sample_data\\pandora\\Pandora135s1_ManhattanNY-CCNY_L2_rfus5p1-8.pkl',
     'datas\\sample_data\\pandora\\Pandora135s1_ManhattanNY-CCNY_L2_rnvs3p1-8.parquet',
     'datas\\sample_data\\pandora\\Pandora135s1_ManhattanNY-CCNY_L2_rnvs3p1-8.pkl',
     'datas\\sample_data\\pandora\\Pandora135s1_ManhattanNY-CCNY_L2_rout2p1-8.parquet',
     'datas\\sample_data\\pandora\\Pandora135s1_ManhattanNY-CCNY_L2_rout2p1-8.pkl',
     'datas\\sample_data\\pandora\\Pandora135s1_ManhattanNY-CCNY_L2_rsus1p1-8.parquet',
     'datas\\sample_data\\pandora\\Pandora135s1_ManhattanNY-CCNY_L2_rsus1p1-8.pkl',
     'datas\\sample_data\\pandora\\Pandora135s1_ManhattanNY-CCNY_L2_rwvt1p1-8.parquet',
     'datas\\sample_data\\pandora\\Pandora135s1_ManhattanNY-CCNY_L2_rwvt1p1-8.pkl',
     'datas\\sample_data\\pandora\\Pandora156s1_HamptonVA-HU_L2_rfus5p1-8.parquet',
     'datas\\sample_data\\pandora\\Pandora156s1_HamptonVA-HU_L2_rfus5p1-8.pkl',
     'datas\\sample_data\\pandora\\Pandora156s1_HamptonVA-HU_L2_rnvs3p1-8.parquet',
     'datas\\sample_data\\pandora\\Pandora156s1_HamptonVA-HU_L2_rnvs3p1-8.pkl',
     'datas\\sample_data\\pandora\\Pandora156s1_HamptonVA-HU_L2_rout2p1-8.parquet',
     'datas\\sample_data\\pandora\\Pandora156s1_HamptonVA-HU_L2_rout2p1-8.pkl',
     'datas\\sample_data\\pandora\\Pandora156s1_HamptonVA-HU_L2_rsus1p1-8.parquet',
     'datas\\sample_data\\pandora\\Pandora156s1_HamptonVA-HU_L2_rsus1p1-8.pkl',
     'datas\\sample_data\\pandora\\Pandora156s1_HamptonVA-HU_L2_rwvt1p1-8.parquet',
     'datas\\sample_data\\pandora\\Pandora156s1_HamptonVA-HU_L2_rwvt1p1-8.pkl',
     'datas\\sample_data\\pandora\\Pandora255s1_VirginiaBeachVA-CBBT_L2_rout2p1-8.parquet',
     'datas\\sample_data\\pandora\\Pandora255s1_VirginiaBeachVA-CBBT_L2_rout2p1-8.pkl',
     'datas\\sample_data\\pandora\\Pandora255s1_VirginiaBeachVA-CBBT_L2_rwvt1p1-8.parquet',
     'datas\\sample_data\\pandora\\Pandora255s1_VirginiaBeachVA-CBBT_L2_rwvt1p1-8.pkl',
     'datas\\sample_data\\pandora\\Pandora2s1_GreenbeltMD_L2_rfus5p1-8.parquet',
     'datas\\sample_data\\pandora\\Pandora2s1_GreenbeltMD_L2_rfus5p1-8.pkl',
     'datas\\sample_data\\pandora\\Pandora2s1_GreenbeltMD_L2_rnvs3p1-8.parquet',
     'datas\\sample_data\\pandora\\Pandora2s1_GreenbeltMD_L2_rnvs3p1-8.pkl',
     'datas\\sample_data\\pandora\\Pandora2s1_GreenbeltMD_L2_rout2p1-8.parquet',
     'datas\\sample_data\\pandora\\Pandora2s1_GreenbeltMD_L2_rout2p1-8.pkl',
     'datas\\sample_data\\pandora\\Pandora2s1_GreenbeltMD_L2_rsus1p1-8.parquet',
     'datas\\sample_data\\pandora\\Pandora2s1_GreenbeltMD_L2_rsus1p1-8.pkl',
     'datas\\sample_data\\pandora\\Pandora2s1_GreenbeltMD_L2_rwvt1p1-8.parquet',
     'datas\\sample_data\\pandora\\Pandora2s1_GreenbeltMD_L2_rwvt1p1-8.pkl',
     'datas\\sample_data\\pandora\\Pandora32s1_GreenbeltMD_L2_rfus5p1-8.parquet',
     'datas\\sample_data\\pandora\\Pandora32s1_GreenbeltMD_L2_rfus5p1-8.pkl',
     'datas\\sample_data\\pandora\\Pandora32s1_GreenbeltMD_L2_rnvs3p1-8.parquet',
     'datas\\sample_data\\pandora\\Pandora32s1_GreenbeltMD_L2_rnvs3p1-8.pkl',
     'datas\\sample_data\\pandora\\Pandora32s1_GreenbeltMD_L2_rout2p1-8.parquet',
     'datas\\sample_data\\pandora\\Pandora32s1_GreenbeltMD_L2_rout2p1-8.pkl',
     'datas\\sample_data\\pandora\\Pandora32s1_GreenbeltMD_L2_rsus1p1-8.parquet',
     'datas\\sample_data\\pandora\\Pandora32s1_GreenbeltMD_L2_rsus1p1-8.pkl',
     'datas\\sample_data\\pandora\\Pandora32s1_GreenbeltMD_L2_rwvt1p1-8.parquet',
     'datas\\sample_data\\pandora\\Pandora32s1_GreenbeltMD_L2_rwvt1p1-8.pkl',
     'datas\\sample_data\\pandora\\Pandora37s1_HamptonVA_L2_rout2p1-8.parquet',
     'datas\\sample_data\\pandora\\Pandora37s1_HamptonVA_L2_rout2p1-8.pkl',
     'datas\\sample_data\\pandora\\Pandora37s1_HamptonVA_L2_rwvt1p1-8.parquet',
     'datas\\sample_data\\pandora\\Pandora37s1_HamptonVA_L2_rwvt1p1-8.pkl',
     'datas\\sample_data\\pandora\\Pandora80s1_BeltsvilleMD_L2_rfus5p1-8.parquet',
     'datas\\sample_data\\pandora\\Pandora80s1_BeltsvilleMD_L2_rfus5p1-8.pkl',
     'datas\\sample_data\\pandora\\Pandora80s1_BeltsvilleMD_L2_rnvs3p1-8.parquet',
     'datas\\sample_data\\pandora\\Pandora80s1_BeltsvilleMD_L2_rnvs3p1-8.pkl',
     'datas\\sample_data\\pandora\\Pandora80s1_BeltsvilleMD_L2_rout2p1-8.parquet',
     'datas\\sample_data\\pandora\\Pandora80s1_BeltsvilleMD_L2_rout2p1-8.pkl',
     'datas\\sample_data\\pandora\\Pandora80s1_BeltsvilleMD_L2_rsus1p1-8.parquet',
     'datas\\sample_data\\pandora\\Pandora80s1_BeltsvilleMD_L2_rsus1p1-8.pkl',
     'datas\\sample_data\\pandora\\Pandora80s1_BeltsvilleMD_L2_rwvt1p1-8.parquet',
     'datas\\sample_data\\pandora\\Pandora80s1_BeltsvilleMD_L2_rwvt1p1-8.pkl',
     'datas\\sample_data\\RWP\\w21137.cns',
     'datas\\sample_data\\RWP\\w21138.cns',
     'datas\\sample_data\\RWP\\w21139.cns',
     'datas\\sample_data\\RWP\\w21140.cns',
     'datas\\sample_data\\RWP\\w21141.cns',
     'datas\\sample_data\\RWP\\w21142.cns',
     'datas\\sample_data\\RWP\\w21143.cns']},
    include_package_data=True,
    install_requires=reqs,
)



