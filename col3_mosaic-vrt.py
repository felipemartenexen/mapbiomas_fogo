import os
import glob
from osgeo import gdal
from termcolor import colored

#Variaveis
biomes = ["amazonia"] 
years = [2023] #2019, 2020, 2021, 2022, 2023
satellites = ["l8", "l9", "l89", "l789"] #"l7", "l8", "l9", "l789", "l78", "l89", "l57", "l5"
folder_path_base = "../../../../mnt/Files-Geo/Arquivos/col3_mosaics_landsat_30m"
vrt_folder_base = f"{folder_path_base}/vrt"

def create_vrt_file(file_list, output_vrt_path):
    """Create a VRT file from the list of input files."""
    vrt_options = gdal.BuildVRTOptions(resampleAlg='nearest', addAlpha=False)
    gdal.BuildVRT(output_vrt_path, file_list, options=vrt_options)

def process_biome(biome, year, satellite, folder_path, vrt_folder_path):
    print(colored(f"\nProcessing biome: {biome}, year: {year}, satellite: {satellite}", 'yellow'))

    # Use glob to filter files with the specified satellite, biome, and year
    matching_files = glob.glob(os.path.join(folder_path, f"{satellite}_{biome}_{year}*"))

    # Check if there are matching files to merge
    if not matching_files:
        print(f"No files found for {biome}, year {year}, satellite {satellite} with the specified satellite.")
        return

    # Output VRT file name
    output_vrt_name = f"{satellite}_{biome}_{year}.vrt"
    output_vrt_path = os.path.join(vrt_folder_path, output_vrt_name)

    # List to keep track of errors, if any
    errors = []

    # Counter for processed files
    file_count = 0

    # Create the VRT file from the list of matching files
    for file_path in matching_files:
        try:
            # Attempt to open the file
            ds = gdal.Open(file_path)

            # Check if the file was successfully opened
            if ds is None:
                errors.append(f"Warning 1: Can't open {file_path}. Skipping it")
            else:
                # File opened successfully, close the dataset
                ds = None

        except Exception as e:
            # If an exception occurs during file reading, log the error
            errors.append(f"Error: {e} - {file_path}")

        file_count += 1
        print(f"Processed {file_count}/{len(matching_files)} files for {biome}, year {year}, satellite {satellite}.")

    # If there are errors, print them
    if errors:
        print("\nErrors:")
        for error in errors:
            print(error)

    if not errors:
        # If there are no errors, proceed to create the VRT file
        create_vrt_file(matching_files, output_vrt_path)
        print(colored(f"\nVRT file '{output_vrt_name}' created in '{vrt_folder_path}' for {biome}, year {year}, satellite {satellite}.", 'green'))
    else:
        print(colored(f"\nFailed to create VRT file for {biome}, year {year}, satellite {satellite} due to errors.", 'red'))

def main():
    # Loop através dos biomas
    for biome in biomes:
        for year in years:
            for satellite in satellites:
                folder_path = os.path.join(folder_path_base, biome)
                vrt_folder_path = os.path.join(vrt_folder_base, biome)

                # Create the VRT folder if it doesn't exist
                if not os.path.exists(vrt_folder_path):
                    os.makedirs(vrt_folder_path)

                process_biome(biome, year, satellite, folder_path, vrt_folder_path)

if __name__ == "__main__":
    main()