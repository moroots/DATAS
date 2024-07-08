# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 08:04:23 2024

@author: Maurice Roots

"""

from bs4 import BeautifulSoup
import requests
import pandas as pd
import numpy as np

from datetime import datetime as dt
import json
from tqdm import tqdm
from pathlib import Path

import pickle

from concurrent.futures import ThreadPoolExecutor, as_completed

#%%
class Pandora():

    def __init__(self):
        self.data = {}
        self.info = {}
        self.errors = []
        return

    def _get_file(self, filepath):
        filepath = Path(filepath)
        try:
            df, info, header = self._get_data(filepath)
            self.data[filepath.name] = {"data": df, "info": info, "header": header}
            return True
        except Exception as e:
            self.errors.append(e)
            return False
        return self.data

    def _read_pandora_parquet(self, filepath, filters=[], **kwargs):
        """
        Reads a parquet file and applies date filters.

        Parameters:
        filepath (str): Path to the parquet file.
        filters (list): List of filters to apply to the parquet file.
        kwargs (dict): Additional keyword arguments, including 'date_range' for date filtering.

        Returns:
        pd.DataFrame: The filtered DataFrame.
        """
        if "date_range" in kwargs.keys():
            start_date = pd.Timestamp(kwargs["date_range"][0])
            end_date = pd.Timestamp(kwargs["date_range"][1])
            filters.extend([("Column_1", ">=", start_date), ("Column_1", "<=", end_date)])
        return pd.read_parquet(filepath, filters=filters)

    def import_pandora_parquet(self, path, troubleshoot=True, max_workers=2, **kwargs):
        """
        Imports multiple parquet files and applies date filters if specified.

        Parameters:
        path : str, list of str or pathlib object
            Path or list of paths to the parquet files or directories containing parquet files.
        troubleshoot : bool, optional
            Whether to print troubleshooting messages. The default is True.
        max_workers : int, optional
            Maximum number of worker threads to use for parallel file processing. The default is 2.
        **kwargs : dict, optional
            Additional keyword arguments, including 'date_range' for date filtering.

        Returns:
        self
            Data is located in self.data as a dictionary of pandas DataFrames.
        """
        if not isinstance(path, (list, tuple)):
            path = [path]

        all_filepaths = []
        for p in path:
            p = Path(p)
            if not p.exists():
                if troubleshoot:
                    print(f"{str(p)}: Does not exist on this machine...")
                continue

            if p.is_dir():
                all_filepaths.extend([x for x in p.glob("*.parquet") if x.is_file()])
            elif p.is_file():
                all_filepaths.append(p)

        with tqdm(total=len(all_filepaths), desc="Importing Pandora Files") as pbar:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(self._read_and_load_pickles, fp, **kwargs): fp for fp in all_filepaths}
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        filename = futures[future].stem  # Using the stem to avoid extension in the key
                        self.data[filename] = result
                    except Exception as e:
                        if troubleshoot:
                            print(f"Error processing {futures[future]}: {e}")
                        self.errors.append(e)
                    pbar.update(1)

        return self

    def _read_and_load_pickles(self, filepath, **kwargs):
        """
        Reads the parquet file and loads the pickled header and instrument info.

        Parameters:
        filepath (str): Path to the parquet file.
        kwargs (dict): Additional keyword arguments for _read_pandora_parquet.

        Returns:
        dict: Dictionary containing DataFrame, header info, and instrument info.
        """
        df = self._read_pandora_parquet(filepath, **kwargs)

        # Load pickled header and instrument info
        pickle_path = filepath.with_suffix('.pkl')
        with open(pickle_path, 'rb') as f:
            header_info, instrument_info = pickle.load(f)

        # Convert bytes to strings
        header_info = json.loads(header_info.decode('utf-8'))
        instrument_info = json.loads(instrument_info.decode('utf-8'))

        return {"data": df, "info": {"Column_info": header_info, "instrument_info": instrument_info}}


    def import_pandora(self, path, troubleshoot=True, max_workers=2):
        if not hasattr(path, '__iter__') or type(path) is str:
            path = [path]

        all_filepaths = []
        for p in path:
            p = Path(p)
            if not p.exists():
                print(f"{str(p)}: Does not exist on this machine...")
                return

            if p.is_dir():
                all_filepaths.extend([x for x in p.glob("*.txt") if x.is_file()])
            if p.is_file():
                all_filepaths.append(p)

        with tqdm(total=len(all_filepaths), desc="Importing Pandora Files") as pbar:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(self._get_file, fp): fp for fp in all_filepaths}
                for future in as_completed(futures):
                    pbar.update(1)

        return self

    def _get_header(self, filePath):

        with open(filePath, 'r') as f:
            line_num = 0
            header_lines = []
            header = {"columns": {}, "info":{}}
            header_finished = False

            for line in f:
                line_num += 1
                if line_num == 100:
                    print("Found no Header Endings...")
                    break

                if (header_finished is True) and (line_num == header_lines[-1]+1):
                    num_cols = len(line.split(" "))
                    break

                if any(char in line for char in ["---"]):
                    header_lines.append(line_num)

                elif header_finished is False:
                    line_info = line.split(":")
                    key = line_info[0].replace(" ", "_")
                    value = ' '.join(line_info[1:]).lstrip().rstrip("\n")
                    if len(header_lines) < 1:
                        header["info"][key] = value
                    else:
                        header["columns"][key] = value

                if len(header_lines) == 2:
                    header_finished = True

        return header, header_lines, num_cols

    def _format_info(self, header, data):
        start_time = data["Column_1"].iloc[0][:15]
        end_time = data["Column_1"].iloc[-1][:15]

        header["info"]["Start_time"] = str(dt.strptime(start_time, '%Y%m%dT%H%M%S'))
        header["info"]["End_time"] = str(dt.strptime(end_time, '%Y%m%dT%H%M%S'))

        header_json = json.dumps(header["columns"])
        header_binary = bytes(header_json, 'utf-8')

        instrument_json = json.dumps(header["info"])
        instrument_binary = bytes(instrument_json, 'utf-8')

        header["info"]["Column_info"] = header_binary
        header["info"]["instrument_info"] = instrument_binary

        return header["info"]

    def _format_data(self, df):
        df["Column_1"] = df["Column_1"].apply(lambda x: x[:15]).apply(lambda x: dt.strptime(x, '%Y%m%dT%H%M%S'))
        for column in df.columns:
            if column != "Column_1":
                df[column] = df[column].apply(pd.to_numeric)
        return df

    def _read_data(self, filePath):

        header, header_lines, num_cols = self._get_header(filePath)

        columns = [f"Column_{i}" for i in range(1, num_cols+1)]

        df = pd.read_csv(filePath, sep=" " , skiprows=header_lines[1], names=columns, low_memory=False, on_bad_lines="skip", float_precision="high", encoding_errors="replace")

        return df, header

    def _get_data(self, filePath):
        df, header = self._read_data(filePath)

        info = self._format_info(header, df)

        df = self._format_data(df)

        return df, info, header


class Pandonia(Pandora):
    def __init__(self):
        super().__init__()
        self.base_url = r"https://data.pandonia-global-network.org"
        self.all_stations = self.list_sites()
        self.products = {"rfuh": "formaldehyde_profile",
                         "rfus": "formaldehyde_column",
                         "rnvh": "NO2_profile",
                         "rnvs": "NO2_column",
                         "rout": "O3_column",
                         "rsus": "SO2_column",
                         "rwvt": "waterVapor_column"}
        return

    def get_page_content(self, url):
        response = requests.get(url)
        response.raise_for_status()
        return response.text

    def filter_a_content(self, soup):
        # Extract text from all 'a' tags, clean it, and filter out unwanted entries
        a_lists = [
            a_tag.text.strip().replace("/", "")
            for a_tag in soup.find_all('a')
            if "\t" not in a_tag.text
        ]
        return [x for x in a_lists if x]

    def list_sites(self):
        content = self.get_page_content(self.base_url)
        soup = BeautifulSoup(content, 'html.parser')
        sites = self.filter_a_content(soup)
        sites.remove('calibrationfiles')
        sites.remove('operationfiles')
        return sites

    def _instruments(self, site: list):
        url = f"{self.base_url}/{site}"
        content = self.get_page_content(url)
        soup = BeautifulSoup(content, 'html.parser')
        instruments = self.filter_a_content(soup)
        return [x for x in instruments if "Pandora" in x]

    def _filenames(self, site: str, instrument: str, version="L2"):
        url = f"{self.base_url}/{site}/{instrument}/{version}"
        try:
            content = self.get_page_content(url)
            soup = BeautifulSoup(content, 'html.parser')
            filenames = self.filter_a_content(soup)

        except Exception as e:
            self.errors.append(e)
            return []

        return filenames


    def download_data(self, sites, dirpath, products=["rout", "rsus", "rwvt", "rnvs", "rfus"], version="L2", parquet=True, max_workers=2):
        dirpath = Path(dirpath)

        if not dirpath.is_dir():
            response = input(f"\n {dirpath} is not found on this machine. Would you like to create it? (Y or N) \n")
            if response.lower() == "y":
                dirpath.mkdir(parents=True, exist_ok=True)
            else:
                return

        total_files = 0
        file_info = []

        for site in sites:
            instruments = self._instruments(site)
            for instrument in instruments:
                filenames = self._filenames(site, instrument)
                filtered_filenames = [x for x in filenames if any(sub in x for sub in products)]
                total_files += len(filtered_filenames)
                for filename in filtered_filenames:
                    file_info.append((site, instrument, filename))

        def download_file(site, instrument, filename):
            url = f"{self.base_url}/{site}/{instrument}/{version}/{filename}"
            try:
                self._download(url, dirpath / filename, parquet=parquet)
            except Exception as e:
                self.errors.append(e)

        with tqdm(total=total_files, desc="Downloading files") as pbar:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(download_file, site, instrument, filename) for site, instrument, filename in file_info]
                for future in as_completed(futures):
                    pbar.update(1)

        return

    def _to_parquet(self, filepaths, savepath, max_workers=2):
        dirpath = Path(savepath)

        if not dirpath.is_dir():
            response = input(f"\n {dirpath} is not found on this machine. Would you like to create it? (Y or N) \n")
            if response.lower() == "y":
                dirpath.mkdir(parents=True, exist_ok=True)
            else:
                return []

        def process_file(filepath):
            savenames = []
            try:
                df, info, header = self._get_data(filepath)
                savename = dirpath / filepath.name.replace(".txt", ".parquet")
                pickle_savename = dirpath / filepath.name.replace(".txt", ".pkl")

                # Save the temp information as a pickle file
                temp = (info["Column_info"], info["instrument_info"])
                with open(pickle_savename, 'wb') as f:
                    pickle.dump(temp, f)

                # Save the DataFrame to a parquet file
                df.to_parquet(path=savename)
                savenames.append(savename)
            except Exception as e:
                self.errors.append(e)
            return savenames

        all_savenames = []

        with tqdm(total=len(filepaths), desc="Processing files to Parquet") as pbar:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(process_file, filepath) for filepath in filepaths]
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        all_savenames.extend(result)
                    except Exception as e:
                        self.errors.append(e)
                    pbar.update(1)

        return all_savenames



    def _download(self, file_url, filepath, parquet=True):
        filepath = Path(filepath)
        html = requests.get(file_url).content
        with open(filepath, "wb") as f:
            f.write(html)

        # if parquet == True:
        #     self._to_parquet(filepath, filepath.parent)

        return html

    @staticmethod
    def _download_specific(file_url, num_lines=-5000):
        lines = requests.get(file_url).text.split("\n")[num_lines:]
        df = pd.DataFrame(lines)
        df.to_parquet(f"temp_{file_url.split('/')[-1]}.txt")
        return

#%%
if __name__ == "__main__":

    # sites=["HamptonVA", "HamptonVA-HU", "GreenbeltMD",'ManhattanNY-CCNY',
    #        'BeltsvilleMD', 'VirginiaBeachVA-CBBT']

    pandora = Pandonia()

    # dirpath = r"C:\Users\meroo\OneDrive - UMBC\Research\Analysis\DATAS\JCRD\data\pandora\test"
    # dirpath = r"C:\Users\Magnolia\OneDrive - UMBC\Research\Analysis\DATAS\JCRD\data\pandora\test"
    # print(pandora.all_stations)
    # pandora.download_data(sites, dirpath, parquet=False, max_workers=2)


#%%
    # dirpath = Path(r"C:\Users\meroo\OneDrive - UMBC\Research\Analysis\DATAS\JCRD\data\pandora\test")
    # dirpath = Path( r"C:\Users\Magnolia\OneDrive - UMBC\Research\Analysis\DATAS\JCRD\data\pandora\test")
    dirpath = Path(r"C:\Users\meroo\OneDrive - UMBC\Research\Analysis\DATAS\JCRD\data\pandora")


    test = pandora.import_pandora_parquet(dirpath, date_range=("2024-07-03", "2024-07-06"))



#%% 

    data = test.data
    
#%% 

# 39: Total Vertical Column
# 4: Solar Zenith Angle
# 5: Solar Azimuth Angle 0deg = North, increaseing clockwise
# 6: Lunar Zenith Angle
# 7: Lunar Azimuth Angle 0deg = North, increaseing clockwise
# 26: Atmospheric Variability
# 30: L1 data quality flag, 0=assured high quality, 1=assured medium quality, 2=assured low quality, 10=not-assured high quality, 11=not-assured medium quality, 12=not-assured low quality
# 33: L2Fit data quality flag, 0=assured high quality, 1=assured medium quality, 2=assured low quality, 10=not-assured high quality, 11=not-assured medium quality, 12=not-assured low quality
# 36: L2 data quality flag for formaldehyde, 0=assured high quality, 1=assured medium quality, 2=assured low quality, 10=not-assured high quality, 11=not-assured medium quality, 12=not-assured low quality, 20=unusable high quality, 21=unusable medium quality, 22=unusable low quality
# 43: Total uncertainty of formaldehyde total vertical column amount [moles per square meter], -1=cross section is zero in this wavelength range, -3=spectral fitting was done, but no independent uncertainty could be retrieved, -5=no independent uncertainty input was given, -6=no common uncertainty input was given, -7=not given since method "MEAS" was chosen, -8=not given, since not all components are given




    for key in data.keys():
        print(data[key]["info"]["Column_info"]["Column_43"])
    
    #%% 
    




































































