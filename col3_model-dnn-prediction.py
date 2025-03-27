#!/usr/bin/env python
# coding: utf-8

#ee.Authenticate()
# -- PARÂMETROS BÁSICOS --
# ------------------------------------------------
version = 2
biome = 'amazonia'
region = '9'
satellite = 'l89' # l8 l9 l89 l789 l78 l5 l57
folder = '../dados'# pasta principal onde fica os dados
folder_modelo = '../../../../mnt/Files-Geo/Arquivos/modelos_col3'
folder_mosaic = f'../../../../mnt/Files-Geo/Arquivos/col3_mosaics_landsat_30m/vrt/{biome}'
years = [2023] 
#2013,2014,2015,2016,2017,2018,2019,2020,2021,2022,2023
sulfix = '_6'

# Set the GPU memory growth to limit memory usage
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# FAZ A PREDIÇÃO

# Reshape into a single vector of pixels for data classify
def reshape_sigle_vector(data_classify):
    data_classify_vector = data_classify.reshape([data_classify.shape[0] * data_classify.shape[1], data_classify.shape[2]])
    return data_classify_vector

def classify(data_classify_vector):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.50)
    with tf.Session(graph=graph, config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

        # restaura o valor das variáveis
        saver.restore(sess, f'{folder_modelo}/col3_{biome}_r{region}_v{version}_rnn_lstm_ckpt')

        # faz a classificação - divide em cinco partes para evitar "Kernel memory"
        output_data_classify0 = outputs.eval({x_input: data_classify_vector[:4000000, bi]})
        output_data_classify1 = outputs.eval({x_input: data_classify_vector[4000000:8000000, bi]})
        output_data_classify2 = outputs.eval({x_input: data_classify_vector[8000000:12000000, bi]})
        output_data_classify3 = outputs.eval({x_input: data_classify_vector[12000000:, bi]})

        output_data_classify = np.concatenate([
            output_data_classify0,
            output_data_classify1,
            output_data_classify2,
            output_data_classify3
        ])
    # Libera recursos da sessão TensorFlow
    tf.keras.backend.clear_session()
    return output_data_classify

def reshape_image_output(output_data_classified, data_classify):
    output_image_data = output_data_classified.reshape(
        [data_classify.shape[0], data_classify.shape[1]])
    return output_image_data

def filter_spacial(output_image_data):
    binary_image = output_image_data > 0

    # Remove small white regions
    open_image = ndimage.binary_opening(
        binary_image, structure=np.ones((3, 3)))
    # Remove small black hole
    close_image = ndimage.binary_closing(open_image, structure=np.ones((9, 9)))
    return close_image

def convert_to_raster(dataset_classify, image_data_scene, output_image_name):
    cols = dataset_classify.RasterXSize
    rows = dataset_classify.RasterYSize
    ds = dataset_classify

    driver = gdal.GetDriverByName('GTiff')
    outDs = driver.Create(output_image_name, cols, rows, 1, gdal.GDT_Float32)
    outDs.GetRasterBand(1).WriteArray(image_data_scene)

    # follow code is adding GeoTransform and Projection
    geotrans = ds.GetGeoTransform()  # get GeoTransform from existed 'data0'
    proj = ds.GetProjection()  # you can get from a existed tif or import
    outDs.SetGeoTransform(geotrans)
    outDs.SetProjection(proj)
    outDs.FlushCache()
    outDs = None

def render_classify(dataset_classify):
    data_classify = convert_to_array(dataset_classify)

    data_classify_vector = reshape_sigle_vector(data_classify)
    # data_classify_vector = add_server_features(data_classify_vector)

    output_data_classified = classify(data_classify_vector)

    output_image_data = reshape_image_output(
        output_data_classified, data_classify)

    return filter_spacial(output_image_data)

def read_grid_landsat():

    grid = ee.FeatureCollection(f'users/geomapeamentoipam/AUXILIAR/grid_regions/grid-{biome}-{region}')

    grid_features = grid.getInfo()['features']
    return grid_features

# clip da imagem pelo grid no limite do bioma
def clip_image_by_grid(geom, image, output):
    with rasterio.open(image) as src:
        print(f"Image path: {image}")
        print(f"Output path: {output}")
        try:
            out_image, out_transform = mask(
                src, geom, crop=True, nodata=np.nan, filled=True)
        except ValueError as e:
            print(f'Skipping image: {image} - {str(e)}')
            return
    out_meta = src.meta.copy()

    # save the resulting raster
    out_meta.update({"driver": "GTiff",
                     "height": out_image.shape[1],
                     "width": out_image.shape[2],
                     "transform": out_transform})

    with rasterio.open(output, "w", **out_meta) as dest:
        dest.write(out_image)


grid_landsat = read_grid_landsat()
start_time = time.time()

for year in years:
    vrt_NBR = f'{folder_mosaic}/{satellite}_{biome}_{year}.vrt'
    #os.system(f'gdalbuildvrt {vrt_NBR} {files_NBR}')    

    input_scenes = []
    total_scenes_done = 0
    for grid in grid_landsat:

        orbit = grid['properties']['ORBITA']
        point = grid['properties']['PONTO']

        output_image_name = f'{folder}/image_col3_{biome}_r{region}_v{version}_{orbit}_{point}_{year}.tif'

        if not os.path.isfile(output_image_name):

            geometry_cena = [grid['geometry']]
            feature_grid = ee.Feature(grid)
            area_grid = feature_grid.area().divide(1000 * 1000).getInfo() # km²

            #clip da imagem de NBR
            print(colored(f'Clipping image mosic for scene {orbit}/{point}', 'cyan'))
            NBR_clipped = f'{folder}/image_mosaic_col3_{biome}_r{region}_v{version}_{orbit}_{point}_clipped_{year}.tif'

            try:
                clip_image_by_grid(geometry_cena, vrt_NBR, NBR_clipped)
            except ValueError as e:
                print(colored(f'Error clipping image: {NBR_clipped} - {str(e)}', 'red'))
                print(f'Full path to the clipped image: {os.path.abspath(NBR_clipped)}')

            images = []
            dataset_classify = None
            try:
                image = NBR_clipped

                if os.path.isfile(image):

                    dataset_classify = load_image(image)
                else:
                    print(colored(f'Image not found: {image}', 'red'))
            except:
                print(colored(f'Image not found: {NBR_clipped}', 'red'))

            # if dataset_classify:
            #     try:
            #         image_data = render_classify(dataset_classify)
            #         convert_to_raster(dataset_classify, image_data, output_image_name)
            #     except:
            #         print(colored(f'Erro classify image: {image}', 'red'))

            if dataset_classify:
                try:
                    image_data = render_classify(dataset_classify)
                    convert_to_raster(dataset_classify, image_data, output_image_name)
                except Exception as e:
                    print(colored(f'Error during classification of image: {image}', 'red'))
                    print(f'Error details: {str(e)}')
                    continue  # Skip to the next iteration in case of an error

        total_scenes_done += 1

        if os.path.isfile(output_image_name):
            input_scenes.append(output_image_name)

            print(colored(f'Done in {year}, {total_scenes_done} scenes of {len(grid_landsat)} scenes ({3:.2f}% complete)'.format(total_scenes_done/len(grid_landsat)*100), 'green'))

        # ----------------------------------------------------

    if len(input_scenes) > 0:
        input_scenes = " ".join(input_scenes)

        image_name = f"queimada_{biome}_{satellite}_v{version}_region{region}_{year}{sulfix}"
        output_image = f"{folder}/{image_name}.tif"

        print(colored('Merging all scenes', 'yellow'))
        os.system(f'gdal_merge.py -n 0 -co COMPRESS=PACKBITS -co BIGTIFF=YES -of gtiff {input_scenes} -o {output_image}')

        os.system(f'gsutil -m cp {output_image} gs://tensorflow-fire-cerrado1/result_classified_colecao3/{biome}')

        outputAssetID = f'projects/ee-geomapeamentoipam/assets/MAPBIOMAS_FOGO/COLECAO_3/{biome.upper()}/{image_name}'
        bucket = f'gs://tensorflow-fire-cerrado1/result_classified_colecao3/{biome}/{image_name}.tif'

        os.system(f'earthengine upload image --asset_id={outputAssetID} {bucket}')

        os.system(f'rm -rf {folder}/image_*')

        print(colored(f'Done {year}', 'green'))

        elapsed_time = time.time() - start_time
        print(colored('Spent time: {0}'.format(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))), 'yellow'))

print(colored('Done all.', 'green'))