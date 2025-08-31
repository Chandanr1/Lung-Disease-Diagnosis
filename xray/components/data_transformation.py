import os 
import sys
from typing import Tuple

import joblib
from torch.utils.data import DataLoader, dataset
from torchvision import  transforms
from torchvision.datasets import ImageFolder
from xray.entity.config_entity import DataTransformationConfig

from xray.entity.artifacts_entity import (DataIngestionArtifact, DataTransformationArtifact)
from xray.exception import XRayException
from xray.logger import logging

class DataTransformation:
    def __init__(self, data_transformation_config: DataTransformationConfig ,data_ingestion_artifact: DataIngestionArtifact):
            self.data_transformation_config = data_transformation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.batch_size = batch_size

    def transforming_training_data(self) ->  transforms.Compose:
        try:
            logging.info(f"Applying data transformations on training and testing data")
            train_transform : transforms.Compose = transforms.Compose([
              transforms.Resize(self.data_transformation_config.RESIZE),
              transforms.CenterCrop(self.data_transformation_config.CENTER_CROP),
              transforms.ColorJitter( **self.data_transformation_config.color_jitter_transforms),
              transforms.RandomHorizontalFlip(),
              transforms.RandomRotation(self.data_transformation_config.RANDOM_ROTATION),
              transforms.ToTensor(),
              transforms.Normalize(**self.data_transformation_config.normalize_transforms),
            ])
            test_data_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            logging.info(f"Data transformations applied on training and testing data")
            return train_transform
        except Exception as e:
            raise XRayException(e, sys)
    
    def transforming_testing_data(self) -> transforms.Compose:
        logging.info(f"Applying data transformations on testing data")
        try:
            test_transform : transforms.Compose = transforms.Compose([
                transforms.Resize(self.data_transformation_config.RESIZE),
                transforms.CenterCrop(self.data_transformation_config.CENTER_CROP),
                transforms.ToTensor(),
                transforms.Normalize(**self.data_transformation_config.normalize_transforms),
            ])
            return test_transform
        except Exception as e:
            raise XRayException(e, sys)
    
    def data_loader(self, train_transform: transforms.Compose, test_transform: transforms.Compose) -> Tuple[DataLoader, DataLoader]:
        try:
            logging.info(f"Creating data loaders")
            train_data : Dataset = ImageFolder(
                os.path.join(self.data_ingestion_artifact.train_file_path),
                transform=train_transform
            )
            test_data : Dataset = ImageFolder(
                os.path.join(self.data_ingestion_artifact.test_file_path),
                transform=test_transform
            )
            logging.info(f"Data sets created")
            train_loader : DataLoader = DataLoader(
                train_data , **self.data_transformation_config.data_loader_params
            )
            test_loader : DataLoader = DataLoader(
                test_data , **self.data_transformation_config.data_loader_params
            )
            logging.info(f"Data loaders created")
            return train_loader, test_loader
        except Exception as e:
            raise XRayException(e, sys)
    def initiate_data_transformation(self) -> DataTransformationArtifact:
        logging.info(f"Starting data transformation")
        try:
            train_transform : transforms.Compose = self.transforming_training_data()
            test_transform : transforms.Compose = self.transforming_testing_data()
            os.makedirs(self.data_transformation_config.artifact_dir, exist_ok=True)
            joblib.dump(
                train_transform,
                self.data_transformation_config.train_transforms_file
            )
            joblib.dump(
                test_transform,
                self.data_transformation_config.test_transforms_file
            )

            train_loader, test_loader = self.data_loader(train_transform, test_transform)
            data_transformation_artifact : DataTransformationArtifact = DataTransformationArtifact(
                transformed_train_object = train_loader,
                transformed_test_object = test_loader,
                train_transform_file_path = self.data_transformation_config.train_transforms_file,
                test_transform_file_path = self.data_transformation_config.test_transforms_file
            )
            logging.info(f"Data transformation completed")
            return data_transformation_artifact
        except Exception as e:
            raise XRayException(e, sys)
