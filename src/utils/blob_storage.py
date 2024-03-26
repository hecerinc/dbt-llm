from azure.storage.blob import BlobClient
import logging

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s: %(message)s')
logger = logging.getLogger('blob_storage')
logger.setLevel(logging.INFO)

def read_blob(connection_string: str, container: str, path: str, read_string_content: bool = False):
    try:
        blob_client = BlobClient.from_connection_string(connection_string, container, path)

        if read_string_content:
            content = blob_client.download_blob(encoding='UTF-8')
        else:
            content = blob_client.download_blob()
        data = content.readall()

        return data
    except Exception as e:
        logger.exception(f'Failed reading blob: {path}')
        raise e


def write_blob(connection_string: str, container: str, path: str, data: str, overwrite: bool):
    try:
        blob_client = BlobClient.from_connection_string(connection_string, container, path)
        blob_client.upload_blob(data, overwrite=overwrite)
    except Exception as e:
        logger.exception('Failed writing to blob')
        raise e


