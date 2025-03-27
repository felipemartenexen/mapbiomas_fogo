from __future__ import print_function
import datetime
import time
import httplib2
import os
import sys
import json
from termcolor import colored
from apiclient import discovery
import oauth2client
from oauth2client import client
from oauth2client import tools
from oauth2client.service_account import ServiceAccountCredentials

from logbook import Logger, FileHandler, StreamHandler

log = Logger('copy-google-drive-folder')
start = time.time()

####PARAMETERS
# Variável para controlar a opção de copiar todas as imagens ou apenas as que não existem na pasta de destino
#copy_all_images = True  # True se desejar copiar todas as imagens ou como 
copy_all_images = False  # False se quiser copiar apenas as imagens que não existem na pasta de destino
biomes = ["amazonia"]
satellites = ['l789'] #'l7', 'l8', 'l9', 'l789', 'l78', 'l89', 'l57', 'l5'
#sufix = f'{satellite}_{biome}_'
#inserir o link da pasta:
source_folder_id = '1uNPlwGlwdD3R7vyt4Vq5Jp3Jg2B7xW7F'

try:
    import argparse

    flags = argparse.ArgumentParser(parents=[tools.argparser])
    # Adicionando nossos requisitos específicos de linha de comando
    flags.add_argument('--log-dir', '-l', type=str, help='Local para salvar os arquivos de log', default='/tmp')
    flags.add_argument('--log-level', type=str, help='Escolha o nível de log', default='INFO')
    args = flags.parse_args()

except ImportError:
    flags = None

# Função para exibir o progresso
def track_progress(file_counter, total_files):
    progress_percentage = (file_counter / max(total_files, 1)) * 100
    print(colored("Progresso: {:.2f}%".format(progress_percentage), 'green'))
    
# Caminho das Credenciais
SCOPES = 'https://www.googleapis.com/auth/drive'
CLIENT_SECRET_FILE = '../../dados/credentials/client_secrets.json'
APPLICATION_NAME = 'Copiar Pasta do Google Drive'

def get_credentials():
    """Obtém credenciais válidas do usuário.

    Se nada tiver sido armazenado, ou se as credenciais armazenadas forem inválidas,
    o fluxo do OAuth2 é concluído para obter novas credenciais.

    Retorna:
        Credenciais, as credenciais obtidas.
    """
    # Obtém o diretório home do usuário (como /home/usuario/ no Linux ou C:\Users\usuario\ no Windows)
    home_dir = os.path.expanduser('~')

    # Junta o caminho do diretório de credenciais ao diretório home para formar o caminho completo
    credential_dir = os.path.join(home_dir, '.credentials')

    # Verifica se o diretório de credenciais existe, e se não, cria o diretório
    if not os.path.exists(credential_dir):
        os.makedirs(credential_dir)

    # Junta o caminho completo ao arquivo de credenciais (drive-copy-google-folders.json)
    credential_path = os.path.join(credential_dir, 'drive-copy-google-folders.json')

    # Abre o arquivo client_secret.json e carrega os dados do arquivo em um dicionário
    with open(CLIENT_SECRET_FILE) as f:
        client_secret_data = json.load(f)

        # Extrai a chave privada e o e-mail do cliente do dicionário
        private_key = client_secret_data['private_key']
        private_key_id = client_secret_data['private_key_id']
        client_email = client_secret_data['client_email']
        client_id = client_secret_data['client_id']

    # Cria as credenciais do serviço usando a chave privada e o e-mail do cliente
    credentials = ServiceAccountCredentials.from_json_keyfile_dict({
        "private_key": private_key,
        "private_key_id": private_key_id,
        "client_email": client_email,
        "client_id": client_id,
        "type": "service_account",
        "project_id": "my-project-id",  # Substitua pelo ID do seu projeto
    }, SCOPES)

    # Verifica se as credenciais são inválidas (por exemplo, se a chave privada ou o e-mail estão faltando)
    if credentials.invalid:
        raise ValueError("Credenciais inválidas")

    # Retorna as credenciais obtidas para serem usadas nas chamadas à API do Google Drive
    return credentials

# Função para copiar arquivos para um bioma específico
def copy_files_for_biome(biome, sufix, source_folder_id, target_folder_path, copy_all_images, credentials, http, drive_service):
    
    log.info(f"Processando bioma: {biome}")
    log.info(f"Caminho da pasta de destino: {target_folder_path}")

    files = drive_service.files()
    request = files.list(
        q="'{}' in parents".format(source_folder_id),
        fields="nextPageToken, files(id, name, mimeType)"
    )

    total_files = 0
    file_counter = 0

    print(colored("Iniciando loop de arquivos", 'yellow'))

    while request is not None:
        file_page = request.execute(http=http)
        total_files += len(file_page.get('files', []))

        for this_file in file_page['files']:
            file_counter += 1

            if this_file['name'].startswith(sufix):
                print(colored(u"#== Processando arquivo {} {} (Arquivo número {}).".format(
                    this_file['mimeType'],
                    this_file['name'],
                    file_counter
                ), 'yellow'))

                if this_file['mimeType'] != 'application/vnd.google-apps.folder':
                    file_path = os.path.join(target_folder_path, this_file['name'])

                    if not os.path.exists(file_path) or copy_all_images:
                        print(colored("Copiando arquivo para: {}".format(file_path), 'green'))
                        request_file = files.get_media(fileId=this_file['id'])
                        with open(file_path, 'wb') as f:
                            f.write(request_file.execute())
        track_progress(file_counter, total_files)

        total_files = max(total_files, file_counter)
        request = files.list_next(request, file_page)

    print(colored(f"Término do loop de arquivos para o bioma: {biome}", 'yellow'))
    print(colored(f"Cópia concluída para o bioma: {biome}", 'green'))

def main():
    """
    Copia uma pasta do Google Drive para um servidor local
    """

    log_filename = os.path.join(
        args.log_dir,
        'copy-google-drive-folder-{}.log'.format(os.path.basename(time.strftime('%Y%m%d-%H%M%S')))
    )

    # Registrando alguns manipuladores de registro para arquivos de log e saída padrão
    log_handler = FileHandler(
        log_filename,
        mode='w',
        level=args.log_level,
        bubble=True
    )
    stdout_handler = StreamHandler(sys.stdout, level=args.log_level, bubble=True)

    # Iniciando os manipuladores de log para a saída padrão e o arquivo de log
    with stdout_handler.applicationbound():
        with log_handler.applicationbound():
            # Registrando a mensagem de início da cópia da pasta do Google Drive
            log.info("Iniciando a cópia da pasta do Google Drive")
            log.info("Iniciando em {}".format(time.strftime('%l:%M%p %Z em %d de %b de %Y')))

            # Obtendo as credenciais para autenticação no Google Drive
            credentials = get_credentials()
            http = credentials.authorize(httplib2.Http())
            drive_service = discovery.build('drive', 'v3', http=http)

            # Iterando sobre a lista de biomas
            for satellite in satellites:
                # Iterando sobre a lista de biomas
                for biome in biomes:
                    # Caminho da pasta de destino no servidor local
                    target_folder_path = f'../../../../mnt/Files-Geo/Arquivos/col3_mosaics_landsat_30m/{biome}'
                    sufix = f'{satellite}_{biome}_'
                    # Chamando a função para copiar arquivos para o bioma atual
                    copy_files_for_biome(biome, sufix, source_folder_id, target_folder_path, copy_all_images, credentials, http, drive_service)

            print(colored(("Tempo de execução: {}".format(str(datetime.timedelta(seconds=(round(time.time() - start, 3)))))),'green'))

if __name__ == '__main__':
    # Executando a função principal do programa
    main()
