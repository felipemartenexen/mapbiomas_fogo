import cv2
import os
import glob

# Carregar imagens e rótulos
version = 4
regions = ['8'] #'1', '2', '3', '4', '5'
biomes = ["amazonia"]
satellites = ["l89"]  # 'l8' 'l9' 'l78' 'l89' 'l57' 'l5'
base_folder = '../../../../mnt/Files-Geo/Arquivos/amostras_col4/'

# Loop pelos biomas
for biome in biomes:
    # Loop pelos satélites
    for satellite in satellites:
        # Loop pelas regiões
        for region in regions:
            folder = os.path.join(base_folder, f'{biome}_r{region}/')  # pasta principal onde os dados estão localizados

            # Create the folder if it doesn't exist
            os.makedirs(folder, exist_ok=True)

            images_train_test = [
                f'train_test_fire_nbr_{biome}_r{region}_{satellite}_v{version}_*.tif'
            ]

            for index, images in enumerate(images_train_test):
                # Copiar as imagens de treino e teste do bucket para o local
                os.system(f'gsutil -m cp gs://tensorflow-fire-cerrado1/images_train_test_colecao4/{biome}_r{region}/{images} {folder}/')

                images_name = glob.glob(f'{folder}/{images}')