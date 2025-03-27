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
version = 5
collection = 1
start_region = 1
end_region = 9
start_year= 2025
end_year = 2025
start_mes = 2
end_mes = 2
only_band_annual_data = True
assetIdFrom = 'users/geomapeamentoipam/COLECAO_FOGO_SENTINEL/CLASSIFICACAO'
assetIdTo = 'users/geomapeamentoipam/Colecao_fogo_sentinel_mensal_v{0}_amazonia'.format(version)
biomes = ['AMAZONIA'] #'AMAZONIA', 'CAATINGA', 'CERRADO', 'MATA_ATLANTICA', 'PAMPA', 'PANTANAL
#-----------------------------------------------------------------------------------

names = {
    'AMAZONIA': 'Amazônia',
    'CAATINGA': 'Caatinga',
    'CERRADO': 'Cerrado',
    'MATA_ATLANTICA': 'Mata Atlântica',
    'PAMPA': 'Pampa',
    'PANTANAL': 'Pantanal'
}

def getLandsat(year):
    if (year>=2013):
        landsat = 8
    elif year >= 1999 and year <= 2012:
        landsat = 57
    else:
        landsat = 5
    return landsat

def readBiome(biome):
    biomas = ee.FeatureCollection("users/geomapeamentoipam/AUXILIAR/biomas_IBGE_250mil");
    return biomas.filterMetadata('Bioma', 'equals', biome);

def dataToMilliseconds(date):
    timestamp = time.mktime(time.strptime(date, '%Y-%m'))
    #timestamp = time.mktime(time.strptime(date, '%Y-%m-%d'))
    return int(timestamp) * 1000


def alreadyExists(name, ic):
    return any(x['properties']['system:index'] == name for x in ic)


def addProperties(image, year, biome, name, mes):
   
    properties = {
        'version': version,
        'biome': names[biome],
        'collection': collection,
        'region': region,
        'fireMonth': mes,
        #'landsat': getLandsat(year),
        'name': name,
        'source': 'IPAM',
        'system:time_start': dataToMilliseconds('{0}-{1}'.format(year, mes)),
        'system:time_end': dataToMilliseconds('{0}-{1}'.format(year, mes)),
        'theme': 'Fire',
        'year': year
    }
    
    image = image.set(properties)
    
    return image


def calcArea(image, geometry):
    areaImage = image.select('FireYear').multiply(ee.Image.pixelArea().divide(1e6)) # km²
    areaReducer = areaImage.reduceRegion(
      reducer = ee.Reducer.sum(),
      scale = 30,
      maxPixels = 1e13,
      geometry = geometry,
    ).get('FireYear')

    area = areaReducer.getInfo()

    return image.set('areaKm2', area)


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

for biome in biomes:
    geometry = readBiome(names[biome])
    for region in range(start_region, end_region + 1):
        for year in range(start_year, end_year + 1):
            for mes in range(start_mes, end_mes + 1):    
                landsat = getLandsat(year)
                try:
                    name = '{0}-{1}-r{2}-v{3}-mes{4}'.format(biome, year, region, version, mes)
                    if(not alreadyExists(name, currentImageCollection)):

                        nameImage = 'queimada_{0}_v{1}_region{2}_{3}_{4}'.format(biome.lower(), version, region, year, mes)

                        print(colored('Processing the image "{}".'.format(nameImage), 'cyan'))

                        image = ee.Image('{0}/{1}/{2}'.format(assetIdFrom, biome, nameImage))
                        mask = image.eq(1)
                        image = image.updateMask(mask)
                        image = image.select(
                            ['b1'],
                            ['fire']
                        );
                        image = addProperties(image, year, biome, name, mes)

                        # adiciona uma nova banda com as queimadas do ano. Ou seja, junta as queimadas de todos os meses
                        fireMonth = image.reduce(ee.Reducer.sum()).uint8().rename('FireMonth')
                        fireMonth = fireMonth.where(fireMonth.gte(1), 1) # replace valores >= 1 para 1
                        image = image.addBands(fireMonth)

                        #image = calcArea(image, geometry)

                        # se para expotar apenas uma band com os dados anuais
                        if only_band_annual_data:
                            image = image.select('FireMonth')

                        export(image, name)
                        
                        print(colored('Image "{}" exported with success.'.format(nameImage), 'green'))
                        elapsed_time = time.time() - start_time
                        print(colored('Spent time: {0}'.format(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))), 'yellow'))
                    else:
                        print(colored('Image "{}" already exists in the image collection.'.format(nameImage), 'blue'))
                except Exception as error:
                    print(colored('Failed to export image "{0}". {1}'.format(nameImage, error), 'red'))