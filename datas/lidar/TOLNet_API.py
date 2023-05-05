# -*- coding: utf-8 -*-
"""
Created on Fri May  5 10:24:56 2023

@author: meroo
"""

import requests
import pandas as pd
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.units as munits
import datetime

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
        
class TOLNet:
    
    def __init__(self):
        self.products = self.get_product_types()
        self.file_types = self.get_file_types()
        self.instrument_groups = self.get_instrument_groups()
        self.files = self.get_files_list()
        return
    
    @staticmethod
    def get_product_types():
        return pd.DataFrame(requests.get("https://tolnet.larc.nasa.gov/api/data/product_types").json())
    
    @staticmethod
    def get_file_types():
        return pd.DataFrame(requests.get("https://tolnet.larc.nasa.gov/api/data/file_types").json())
    
    @staticmethod
    def get_instrument_groups():
        return pd.DataFrame(requests.get("https://tolnet.larc.nasa.gov/api/instruments/groups").json())

    @staticmethod
    def get_files_list():
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
                 "product_type_name": "str",
                 "file_type_name": "str",
                 "revision": "int16",
                 "near_real_time": "str",
                 "file_size": "int16",
                 "isAccessible": "bool"
                 }

        i = 1
        url = f"https://tolnet.larc.nasa.gov/api/data/1?order=data_date&order_direction=desc"
        response = requests.get(url).status_code
        data_frames = []
        while response == 200:
            data_frames.append(pd.DataFrame(requests.get(url).json()))
            i += 1
            url = f"https://tolnet.larc.nasa.gov/api/data/{i}?order=data_date&order_direction=desc"
            response = requests.get(url).status_code

        df = pd.concat(data_frames, ignore_index=True)
        df["start_data_date"] = pd.to_datetime(df["start_data_date"])
        df["end_data_date"] = pd.to_datetime(df["end_data_date"])
        df["upload_date"] = pd.to_datetime(df["upload_date"])
        return df.astype(dtypes)
    
    @staticmethod
    def import_json(file_id: int) -> pd.DataFrame:
        url = f"https://tolnet.larc.nasa.gov/api/data/json/{file_id}"
        response = requests.get(url).json()

        data = np.array(response["value"]["data"], dtype=float)
        time = np.array(response["datetime"]["data"])
        alt = np.array(response["altitude"]["data"], dtype=float)

        dataset = pd.DataFrame(data=data, index=time, columns=alt)
        dataset.index = pd.to_datetime(dataset.index)
        return dataset
    
    def import_data_json(self, **kwargs):
        file_info = filter_files(self.files).daterange(**kwargs).instrument_group(**kwargs).product_type(**kwargs).file_type(**kwargs).df
        if file_info.size == self.files.size:
            prompt = input("You are about to download ALL TOLNet JSON files available... Would you like to proceed? (yes | no)")
            if prompt.lower() == "no":
                return
        self.data = {}
        for file_name, file_id in zip(file_info["file_name"], file_info["id"]):
            self.data[file_name] = self.import_json(file_id)
        return self
    
    @staticmethod
    def O3_curtain_colors():

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

    def tolnet_curtains(self, smooth=True, **kwargs):

        fig = plt.figure(figsize=(15, 8))
        ax = plt.subplot(111)
        ncmap, nnorm = O3_curtain_colors()

        for name in self.data.keys():
            X, Y, Z = (self.data[name].index, self.data[name].columns, self.data[name].to_numpy().T,)
            im = ax.pcolormesh(X, Y, Z, cmap=ncmap, norm=nnorm, shading="nearest")

        cbar = fig.colorbar(im, ax=ax, pad=0.01, ticks=[0.001, *np.arange(10, 100, 10), 300])
        cbar.set_label(label='Ozone ($ppb_v$)', size=16, weight="bold")

        if "title" in kwargs.keys():
            plt.title(kwargs["title"], fontsize=18)
        else: plt.title(r"$O_3$ Mixing Ratio Profile ($ppb_v$)", fontsize=20)

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
            df = kwargs["surface"][0]
            dummy = np.ones(len(df))*kwargs["surface"][1]
            ax.scatter(df.index, dummy, c=df, cmap=ncmap, norm=nnorm)

        converter = mdates.ConciseDateConverter()
        munits.registry[datetime.datetime] = converter

        ax.xaxis_date()

        # fonts
        plt.setp(ax.get_xticklabels(), fontsize=16)
        plt.setp(ax.get_yticklabels(), fontsize=16)
        cbar.ax.tick_params(labelsize=16)

        plt.tight_layout()

        if "savefig" in kwargs.keys():
            plt.savefig(f"{kwargs['savefig']}", dpi=600)

        plt.show()

        return

if __name__ == "__main__":
    tolnet = TOLNet()