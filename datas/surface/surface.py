# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 16:56:52 2020

@author: Magnolia

Surface Data

"""

import requests
import pandas as pd
import json


def import_epa_aqs_data(year, molecule):
    # Define the URL for the csv file
    url = f'https://aqs.epa.gov/aqsweb/airdata/daily_{molecule}_by_year.zip'

    # Load the data from the csv file into a Pandas DataFrame
    df = pd.read_csv(url, compression='zip')

    # Filter the data by year
    df = df[df['Year'] == year]

    # Return the filtered DataFrame
    return df

import pandas as pd

def get_parameter_id(molecule_name):
    molecule_name = molecule_name.lower()
    parameter_map = {
        'ozone': '44201',
        'carbon monoxide': '42101',
        'sulfur dioxide': '42401',
        'nitrogen dioxide (NO2)': '42602',
        'pm25': '88101',
        'pm10': '88502'
    }
    return parameter_map.get(molecule_name, molecule_name)

def import_hourly_data(year, molecule):
    """
    Import hourly data from AQS website for a given year and molecule.

    Parameters:
    year (str): Year of data to import.
    molecule (str): Molecule to import data for.

    Returns:
    df (pandas.DataFrame): Hourly data for the specified year and molecule.
    """

    # Create URL for hourly data based on year and molecule
    url = f'https://aqs.epa.gov/aqsweb/airdata/hourly_{get_parameter_id(molecule)}_{year}.zip'

    # Read hourly data CSV from ZIP file on website
    try:
        df = pd.read_csv(url, compression='zip', low_memory=False)
    except Exception as e:
        print(f"Error importing data for {molecule} in {year}: {e}")
        return None

    return df

import pandas as pd
import io
import requests

def import_aqs_data(year, parameter, state):
    url = f"https://aqs.epa.gov/aqsweb/airdata/hourly_{parameter.lower()}_{year}.zip"
    content = requests.get(url).content
    df = pd.read_csv(io.StringIO(content.decode('utf-8')), compression='zip')
    df = df[df['State Name'] == state]
    return df


def filter_by_borders(df, lat_min, lat_max, lon_min, lon_max):
    """
    Filters a DataFrame by latitude and longitude borders.

    Args:
        df: A pandas DataFrame with columns for latitude and longitude.
        lat_min: The minimum latitude value for the border.
        lat_max: The maximum latitude value for the border.
        lon_min: The minimum longitude value for the border.
        lon_max: The maximum longitude value for the border.

    Returns:
        A filtered DataFrame that only includes rows where the latitude
        and longitude fall within the specified borders.
    """
    return df[(df['Latitude'] >= lat_min) & (df['Latitude'] <= lat_max) &
              (df['Longitude'] >= lon_min) & (df['Longitude'] <= lon_max)]

df = import_hourly_data("2016", "Ozone")

#%%

lat_min = 34.0228
lat_max = 41.5145
lon_min = -85.3635
lon_max = -72.7951

# Filter data by borders
filtered_df = filter_by_borders(df, lat_min, lat_max, lon_min, lon_max)


#%%
# from numba import jit

# @jit(nopython=False)
def create_nested_dict(df):
    # Initialize empty dictionary
    nested_dict = {}

    # Group DataFrame by state and site
    grouped_df = df.groupby(['State Name', 'Site Num'])

    # Iterate over groups
    for (state, site), site_df in grouped_df:
        # Create site dictionary using dictionary comprehension
        site_dict = {row['Date Local']: {param: row[param] for param in df.columns if param not in ['State Name', 'County Name', 'Site Num', 'Date Local']} for _, row in site_df.iterrows()}

        # Add site dictionary to state dictionary
        if state not in nested_dict:
            nested_dict[state] = {}

        nested_dict[state][site] = site_dict

    return nested_dict

nested_dictioanry = create_nested_dict(filtered_df)