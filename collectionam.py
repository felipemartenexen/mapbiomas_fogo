#!/usr/bin/env python
# coding: utf-8

import ee
import time
import datetime
from termcolor import colored
import datetime
import time

#ee.Authenticate()
ee.Initialize()

# parâmetros a serem alterados
#--------------------------------
version = 1
collection = 3
start_region = 9
end_region = 9
only_band_annual_data = True
assetIdFrom = 'users/ipam_flp/fire_col3/amazonia_col3_v2'
assetIdTo = f'projects/ee-geomapeamentoipam/assets/MAPBIOMAS_FOGO/COLECAO_3/Colecao3_fogo_v{version}'
biomes = ['AMAZONIA']
#'AMAZONIA', 'CAATINGA', 'CERRADO', 'MATA_ATLANTICA', 'PAMPA', 'PANTANAL'

satellite_years = [
    #{'satellite': 'l5', 'years': [1984,1985,1986,1987,1988,1989,1990,1991,1992,1993,1994,1995,1996,1997,1998]},
    #{'satellite': 'l57', 'years': [1999,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012]},
    #{'satellite': 'l78', 'years': [2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021]},
    #{'satellite': 'l78', 'years': [2022, 2023]},
    {'satellite': 'l89', 'years': [2023]}

]
#-----------------------------------------------------------------------------------

names = {
    'AMAZONIA': 'Amazônia',
    'CAATINGA': 'Caatinga',
    'CERRADO': 'Cerrado',
    'MATA_ATLANTICA': 'Mata Atlântica',
    'PAMPA': 'Pampa',
    'PANTANAL': 'Pantanal'
}

def readBiome(biome):
    biomas = ee.FeatureCollection("users/geomapeamentoipam/AUXILIAR/biomas_IBGE_250mil");
    return biomas.filterMetadata('Bioma', 'equals', biome);

def dataToMilliseconds(date):
    timestamp = time.mktime(time.strptime(date, '%Y-%m-%d'))
    return int(timestamp) * 1000


def alreadyExists(name, ic):
    return any(x['properties']['system:index'] == name for x in ic)


def addProperties(image, year, biome, name):
   
    properties = {
        'version': version,
        'biome': names[biome],
        'collection': collection,
        'region': region,
        'name': name,
        'source': 'IPAM',
        'system:time_start': dataToMilliseconds('{}-01-01'.format(year)),
        'system:time_end': dataToMilliseconds('{}-12-31'.format(year)),
        'theme': 'Fire',
        'year': year
    }
    
    image = image.set(properties)
    return image


def export(image, name):
    task = ee.batch.Export.image.toAsset(
        image = image,
        assetId = '{0}/{1}'.format(assetIdTo, name),
        description = name,
        maxPixels = 1e13,
        scale = 30,
    )
    task.start()


# para verificar quais imagens que já foram exportadas
currentImageCollection = ee.ImageCollection(assetIdTo).getInfo()['features']

start_time = time.time()

for satellite_year in satellite_years:
    satellite = satellite_year['satellite']
    years = satellite_year['years']

    for biome in biomes:
        geometry = readBiome(names[biome])
        for region in range(start_region, end_region + 1):
            for year in years:
                try:
                    name = f'{biome}-{year}-r{region}-v{version}'.format(biome, year, region, version)
                    if(not alreadyExists(name, currentImageCollection)):

                        nameImage = f'queimada_{biome.lower()}_{satellite}_v5_region{region}_{year}'

                        print(colored('Processing the image "{}".'.format(nameImage), 'cyan'))

                        image = ee.Image(f'{assetIdFrom}/{nameImage}')
                        mask = image.eq(1)
                        image = image.updateMask(mask)
                        image = image.select(
                            ['b1'],
                            ['fire']
                        );
                        image = addProperties(image, year, biome, name)

                        # adiciona uma nova banda com as queimadas do ano. Ou seja, junta as queimadas de todos os meses
                        fireYear = image.reduce(ee.Reducer.sum()).uint8().rename('FireYear')
                        fireYear = fireYear.where(fireYear.gte(1), 1) # replace valores >= 1 para 1
                        image = image.addBands(fireYear)

                        #image = calcArea(image, geometry)

                        # se para expotar apenas uma band com os dados anuais
                        if only_band_annual_data:
                            image = image.select('FireYear')

                        export(image, name)
                        
                        print(colored('Image "{}" exported with success.'.format(nameImage), 'green'))
                        elapsed_time = time.time() - start_time
                        print(colored('Spent time: {0}'.format(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))), 'yellow'))
                    else:
                        print(colored('Image "{}" already exists in the image collection.'.format(nameImage), 'blue'))
                except Exception as error:
                    print(colored('Failed to export image "{0}". {1}'.format(nameImage, error), 'red'))


