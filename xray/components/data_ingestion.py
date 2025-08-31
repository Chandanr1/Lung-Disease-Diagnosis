import sys

from xray.cloud_storage.s3_operations import S3Operation
from xray.constant.training_pipeline import *
from xray.entity.artifacts_entity import DataIngestionArtifact
from xray.entity.config_entity import DataIngestionConfig
from xray.exception import XRayException
from xray.logger import logging

class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
            self.data_ingestion_config = data_ingestion_config
            self.s3 = S3Operation()

    def get_data_from_s3(self) -> None:
        try:
            logging.info(f"Downloading data from cloud storage")
            self.s3.sync_folder_from_s3(
                folder=self.data_ingestion_config.data_path,
                bucket_name=self.data_ingestion_config.bucket_name,
                bucket_folder_name=self.data_ingestion_config.s3_data_folder,
            )
            logging.info(f"Data downloaded from cloud storage to local folder")
        except Exception as e:
            raise XRayException(e, sys)
    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        logging.info(f"Starting data ingestion")
        try:
            
            self.get_data_from_s3()
            data_ingestion_artifact : DataIngestionArtifact = DataIngestionArtifact(
                train_file_path=self.data_ingestion_config.train_data_path,
                test_file_path=self.data_ingestion_config.test_data_path,
            )
            logging.info(f"Data ingestion completed")
            return data_ingestion_artifact
        except Exception as e:
            raise XRayException(e, sys)