 # -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 13:32:30 2023

@author: Maurice Roots, Arthur Perng, John T. Sullivna
"""

import requests
import pandas as pd
import numpy as np
import datetime

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.units as munits

from dateutil import tz

from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor, as_completed

#%% Function Space

class filter_files:
    def __init__(self, df):
        self.df = df
        pass
        
    
    def daterange(self, min_date: str = None, max_date: str = None, **kwargs) -> pd.DataFrame:
        try:
            self.df = self.df[(self.df["start_data_date"] >= min_date) & (self.df["start_data_date"] <= max_date)]
        except:
            pass
        return self

    def instrument_group(self, instrument_group: list = None, **kwargs) -> pd.DataFrame:
        try:
            self.df = self.df[self.df["instrument_group_id"].isin(instrument_group)]
        except:
            pass
        return self

    def product_type(self, product_type: list = None, **kwargs) -> pd.DataFrame:
        try:
            self.df = self.df[self.df["product_type_id"].isin(product_type)]
        except:
            pass
        return self

    def file_type(self, file_type: list = None, **kwargs) -> pd.DataFrame:
        try:
            self.df = self.df[self.df["file_type_id"].isin(file_type)]
        except:
            pass
        return self
    
    def processing_type(self, processing_type: list = None, **kwargs) -> pd.DataFrame:
        """
        id 1 = centrally proccessed
        id 2 = in-house 
        id 3 = unprocessed
        """
        
        
        
        
        try:
            processing_type_names = []
            types = tolnet.processing_types
            for process in processing_type:
                processing_type_names.append(
                    list(types["processing_type_name"][types["id"] == process])[0])
                    
            self.df = self.df[self.df["processing_type_name"].isin(processing_type_names)]
        except:
            pass
        return self

class TOLNet:

    def __init__(self):
        self.products = self.get_product_types()
        self.file_types = self.get_file_types()
        self.instrument_groups = self.get_instrument_groups()
        self.processing_types = self.get_processing_types()
        pd.set_option('display.max_colwidth', None)
        # self.files = self.get_files_list()
        return
    
    def print_product_types(self): 
        """
        Prints out all products and their respective IDs.
        """
        print("\n TOLNET product IDs:")
        print(self.products[["id", "product_type_name", "description"]].to_string(index=False), '\n')
        return 
    
    
    def print_file_types(self):
        """
        Prints out all available filetypes and their respective IDs.
        """
        print("\n File Types:")
        print(self.file_types[["id", "file_type_name"]].to_string(index=False), '\n')
        return

    def print_instrument_groups(self):
        """
        Prints out all recorded instrument groups and their respective IDs.
        """
        print("\n Instrument Groups:")
        print(self.instrument_groups[["id", "instrument_group_name", "description"]].to_string(index=False), '\n')
        return
    
    def print_processing_types(self):
        """
        Prints out all present processing types and their respective IDs.
        """
        print("\n Processing Types:")
        print(self.processing_types[["id", "processing_type_name"]].to_string(index=False), '\n')
        return
    
    @staticmethod
    def get_product_types():
        """
        Returns a DataFrame containing all product types. 
        The returned DataFrame contains the columns id, processing_type_name, description, 
        display_order, public, and show_on_graph_page.
        """
        return pd.DataFrame(requests.get("https://tolnet.larc.nasa.gov/api/data/product_types").json()).sort_values(by=["id"])

    @staticmethod
    def get_file_types():
        """
        Returns a DataFrame containing all file types.
        The returned DataFrame contains the columns id, file_type_name, description, display_order, and public.
        """
        return pd.DataFrame(requests.get("https://tolnet.larc.nasa.gov/api/data/file_types").json()).sort_values(by=["id"])

    @staticmethod
    def get_instrument_groups():
        """
        Returns a DataFrame containing all instrument groups.
        The returned DataFrame contains the columns id, instrument_group_name, folder_name, description,
        display_order, current_pi(Principle Investigator), doi, and citation_url.
        """
        return pd.DataFrame(requests.get("https://tolnet.larc.nasa.gov/api/instruments/groups").json()).sort_values(by=["id"])
    
    @staticmethod
    def get_processing_types():
        """
        Returns a DataFrame containing all processing types.
        The returned DataFrame contains the columns id, processing_type_name, description, display_order,
        public, and show_on_graph_page.
        """
        return pd.DataFrame(requests.get("https://tolnet.larc.nasa.gov/api/data/processing_types").json()).sort_values(by=["id"])

    @staticmethod
    def get_files_list(min_date, max_date):
        """
        Parameters
        ----------
        min_date : STR
            The starting date for the query, in YYYY-MM-DD format.
        max_date : STR
            The ending date for the query, in YYYY-MM-DD format.

        Returns
        -------
        A DataFrame containing all files from the TOLNet API that fall between the two provided dates.
        The DataFrame contains each file name as well as various descriptors.
        """
        dtypes = {"row": "int16",
                 "count": "int16",
                 "id": "int16",
                 "file_name": "str",
                 "file_server_location": "str",
                 "author": "str",
                 "instrument_group_id": "int16",
                 "product_type_id": "int16",
                 "file_type_id":"int16",
                 "start_data_date": "datetime64[ns]",
                 "end_data_date":"datetime64[ns]",
                 "upload_date":"datetime64[ns]",
                 "public": "bool",
                 "instrument_group_name": "str",
                 "folder_name": "str",
                 "current_pi": "str",
                 "doi": "str",
                 "citation_url": "str",
                 "product_type_name": "str",
                 "processing_type_name": "str",
                 "file_type_name": "str",
                 "revision": "int16",
                 "near_real_time": "str",
                 "file_size": "int16",
                 "latitude": "int16",
                 "longitude": "int16",
                 "altitude": "int16",
                 "isAccessible": "bool"
                 }

        i = 1
        url = f"https://tolnet.larc.nasa.gov/api/data/1?min_date={min_date}&max_date={max_date}&order=data_date&order_direction=desc"
        response = requests.get(url)
        data_frames = []
        while response.status_code == 200:
            data_frames.append(pd.DataFrame(response.json()))
            i += 1
            url = f"https://tolnet.larc.nasa.gov/api/data/{i}?min_date={min_date}&max_date={max_date}&order=data_date&order_direction=desc"
            response = requests.get(url)

        df = pd.concat(data_frames, ignore_index=True)
        df["start_data_date"] = pd.to_datetime(df["start_data_date"])
        df["end_data_date"] = pd.to_datetime(df["end_data_date"])
        df["upload_date"] = pd.to_datetime(df["upload_date"])
        return df.astype(dtypes)

    def _add_timezone(self, time):
        return [utc.replace(tzinfo=tz.gettz('UTC')) for utc in time]

    def change_timezone(self, timezone: str):
        to_zone = tz.gettz(timezone)

        for key in self.data.keys():
            for filename in self.data[key].keys():
                time = self.data[key][filename].index.to_list()
                self.data[key][filename].index = [t.astimezone(to_zone) for t in time]

        return self

    def _json_to_dict(self, file_id: int) -> pd.DataFrame:
        """
        Parameters
        ----------
        file_id : INT
            The ID of the file to retrieve from the API.

        Returns
        -------
        A dictionary containing the file's ozone values and metadata.
        """
        try:
            url = f"https://tolnet.larc.nasa.gov/api/data/json/{file_id}"
            response = requests.get(url).json()
        except:
            print(f"Error with pulling {file_id}")
        return response
    
    def _unpack_data(self, meta_data):
        """
        Parameters
        ----------
        meta_data : A dictionary of a file's metadata

        Returns
        -------
        A DataFrame containing that same metadata
        """
        df = pd.DataFrame(meta_data["value"]["data"], 
                          columns = meta_data["altitude"]["data"],
                          index = pd.to_datetime(meta_data["datetime"]["data"])
                          )
        return df
    
    def import_data(self, min_date, max_date, **kwargs):
        """
        Parameters
        ----------
        min_date : String
            The starting date to take from. Formatted as YYYY-MM-DD.
        max_date: String
            The ending date to take data from. Formatted as YYYY-MM-DD.

        """ 
        
        # print(f"Retrieving all {productquery} files from {date_start} to {date_end}:")
        
        def process_file(file_name, file_id):
            meta_data = self._json_to_dict(file_id)
            data = self._unpack_data(meta_data)
            data.index = self._add_timezone(data.index.to_list())
            return file_name, meta_data, data
        
        self.files = self.get_files_list(min_date, max_date)
        file_info = filter_files(self.files).daterange(**kwargs).instrument_group(**kwargs).product_type(**kwargs).file_type(**kwargs).processing_type(**kwargs).df
    
        if file_info.size == self.files.size:
            prompt = input("You are about to download ALL TOLNet JSON files available... Would you like to proceed? (yes | no)")
            if prompt.lower() != "yes":
                return
    
        self.data = {}
        self.meta_data = {}
        
        # Use ThreadPoolExecutor for multithreading
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_file = {
                executor.submit(process_file, file_name, file_id): file_name
                for file_name, file_id in zip(file_info["file_name"], file_info["id"])
            }
    
            for future in tqdm(as_completed(future_to_file), total=len(future_to_file)):
                file_name = future_to_file[future]
                try:
                    file_name, meta_data, data = future.result()
                     
                    key = (meta_data["fileInfo"]["instrument_group_name"], 
                           meta_data["fileInfo"]["processing_type_name"])
                    
                    if key not in self.data.keys(): 
                        self.data[key] = {}
                        self.meta_data[key] = {}
                        
                    self.data[key][file_name] = data
                    self.meta_data[key][file_name] = meta_data

                except Exception as e:
                    print(f"Error processing file {file_name}: {e}")
    
        return self

    

    @staticmethod
    def O3_curtain_colors():
        """
        Returns
        -------
        The color scheme used in the O3 curtain plots on the TOLNet website.

        """
        ncolors = [np.array([255,  140,  255]) / 255.,
           np.array([221,  111,  242]) / 255.,
           np.array([187,  82,  229]) / 255.,
           np.array([153,  53,  216]) / 255.,
           np.array([119,  24,  203]) / 255.,
           np.array([0,  0,  187]) / 255.,
           np.array([0,  44,  204]) / 255.,
           np.array([0,  88,  221]) / 255.,
           np.array([0,  132,  238]) / 255.,
           np.array([0,  165,  255]) / 255.,
           np.array([0,  235,  255]) / 255.,
           np.array([39,  255,  215]) / 255.,
           np.array([99,  255,  150]) / 255.,
           np.array([163,  255,  91]) / 255.,
           np.array([211,  255,  43]) / 255.,
           np.array([255,  255,  0]) / 255.,
           np.array([250,  200,  0]) / 255.,
           np.array([255,  159,  0]) / 255.,
           np.array([255,  111,  0]) / 255.,
           np.array([255,  63,  0]) / 255.,
           np.array([255,  0,  0]) / 255.,
           np.array([216,  0,  15]) / 255.,
           np.array([178,  0,  31]) / 255.,
           np.array([140,  0,  47]) / 255.,
           np.array([102,  0,  63]) / 255.,
           np.array([200,  200,  200]) / 255.,
           np.array([140,  140,  140]) / 255.,
           np.array([80,  80,  80]) / 255.,
           np.array([52,  52,  52]) / 255.,
           np.array([0,0,0]) ]

        ncmap = mpl.colors.ListedColormap(ncolors)
        ncmap.set_under([1,1,1])
        ncmap.set_over([0,0,0])
        bounds =   [0.001, *np.arange(5, 110, 5), 120, 150, 200, 300, 600]
        nnorm = mpl.colors.BoundaryNorm(bounds, ncmap.N)
        return ncmap, nnorm

    def tolnet_curtains(self, timezone: int =None, **kwargs):
        """
        Parameters
        ----------
        timezone : INT, optional
            The timezone to display the x-axis times in. Defaults to None.
        **kwargs : TYPE
            Parameters for the plots. The accepted fields are title, xlabel, ylabel,
            xlims, ylims, yticks, surface, sonde.

        Returns
        -------
        A reference to the TOLNet object.

        """
        
        inst_grp_names = self.instrument_groups["instrument_group_name"].to_list()
        
        group_names = np.array(list(self.meta_data.keys()))[:,0]
        group_names = np.unique(group_names)
        
        ncmap, nnorm = self.O3_curtain_colors()
        for key in self.data.keys():
            dates = []
            fig, ax = plt.subplots(1, 1, figsize=(15, 8), layout="tight")
            for filename in self.data[key].keys():
                self.data[key][filename] = self.data[key][filename].fillna(value=np.nan)
                
                X, Y, Z = (self.data[key][filename].index, 
                           self.data[key][filename].columns, 
                           self.data[key][filename].to_numpy().T,)
                           
                im = ax.pcolormesh(X, Y, Z, cmap=ncmap, norm=nnorm, shading="nearest")
                dates.append(X[0].date())
            cbar = fig.colorbar(im, ax=ax, pad=0.01, ticks=[0.001, *np.arange(10, 101, 10), 200, 300])
            cbar.set_label(label='Ozone ($ppb_v$)', size=16, weight="bold")
            converter = mdates.ConciseDateConverter()
            munits.registry[datetime.datetime] = converter

            ax.xaxis_date(timezone)
            dates.sort()
        # fonts
            plt.setp(ax.get_xticklabels(), fontsize=16)
            plt.setp(ax.get_yticklabels(), fontsize=16)
            cbar.ax.tick_params(labelsize=16)
            if "title" in kwargs.keys():
                plt.title(kwargs["title"], fontsize=18)
            else: plt.title(f"$O_3$ Mixing Ratio Profile ($ppb_v$) - {key[0]}, {key[1]} \n {dates[0]} - {dates[-1]}", fontsize=20)
            
            if "ylabel" in kwargs.keys():
                ax.set_ylabel(kwargs["ylabel"], fontsize=18)
            else: ax.set_ylabel("Altitude (km AGL)", fontsize=18)
    
            if "xlabel" in kwargs.keys():
                ax.set_xlabel(kwargs["xlabel"], fontsize=20)
            else: ax.set_xlabel("Datetime (UTC)", fontsize=18)
    
            if "xlims" in kwargs.keys():
                lim = kwargs["xlims"]
                lims = [np.datetime64(lim[0]), np.datetime64(lim[-1])]
                ax.set_xlim(lims)
    
            if "ylims" in kwargs.keys():
                ax.set_ylim(kwargs["ylims"])
    
            if "yticks" in kwargs.keys():
                ax.set_yticks(kwargs["yticks"], fontsize=20)
    
            if "surface" in kwargs.keys():
                X, Y, C = kwargs["surface"]
                ax.scatter(X, Y, c=C, cmap=ncmap, norm=nnorm)
    
            if "sonde" in kwargs.keys():
                X, Y, C = kwargs["sonde"]
                ax.scatter(X, Y, c=C, cmap=ncmap, norm=nnorm)
                
            if "savefig" in kwargs.keys() and kwargs['savefig']:
                plotname = f"{key[0]}_{key[1]}_{dates[0]}_{dates[-1]}.png"
                plotname = plotname.replace(' ', '_').replace('-', '_').replace('\\', '').replace('/', '')
                plt.savefig(f"{plotname}", dpi=350)

        plt.show()

        return self


#%% Example
date_start = "2023-08-08"
date_end = "2023-08-11"
product_IDs = [4]

if __name__ == "__main__":
    tolnet = TOLNet()
    print("Created TOLNET instance")
    
    tolnet.print_product_types()
    tolnet.print_processing_types()
    tolnet.print_instrument_groups()
    tolnet.print_file_types()
    
    data = tolnet.import_data(min_date=date_start, max_date=date_end, product_type=product_IDs)
    data.tolnet_curtains()

#%% 
