import cv2
import os
import glob

# Carregar imagens e rótulos
version = '6_2'
biome = 'amazonia'
regions = [4] #['1', '2', '3', '4', '5']
base_folder = '../../../mnt/Files-Geo/Arquivos/amostras_monitor/'

for region in regions:
    folder = os.path.join(base_folder, f'{biome}_r{region}/')  # pasta principal onde os dados estão localizados

    # Create the folder if it doesn't exist
    os.makedirs(folder, exist_ok=True)

    images_train_test = [
        f'train_test_fire_nbr_{biome}_r{region}_sentinel_v{version}_*.tif'
    ]

    for index, images in enumerate(images_train_test):
        # Copiar as imagens de treino e teste do bucket para o local
        os.system(f'gsutil -m cp gs://tensorflow-fire-cerrado1/images_train_test_sentinel/{biome}_r{region}/{images} {folder}/')

        images_name = glob.glob(f'{folder}/{images}')
