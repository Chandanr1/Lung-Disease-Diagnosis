import os
import sys

import bentoml
import torch
import joblib
import torch.nnfunctional as F
from troch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler, StepLR
from torch.utils.data import Dataset
from tqdm import tqdm
from torch.utils.data import DataLoader

from xray.constants.training_pipeline import *
from xray.entity.artifacts_entity import ( DataTransformationArtifact, ModelTrainerArtifact)
from xray.entity.config_entity import  ModelTrainerConfig
from xray.exception import XRayException
from xray.logger import logging
from xray.ml.model.arch import Net

class ModelTrainer:
    def __init__(self,data_transformation_artifact: DataTransformationArtifact , model_trainer_config: ModelTrainerConfig,
                 ):
            self.model_trainer_config : ModelTrainerConfig = model_trainer_config
            self.data_transformation_artifact: DataTransformationArtifact = (data_transformation_artifact)
            self.model : Module = Net()

    def train(self, optimizer: Optimizer) -> None:
        logging.info(f"Training the model")
        try:
             self.model.train()
             pbar = tqdm(self.data_transformation_artifact.transformed_train_object)
             correct : int = 0
             processed=0
             for batch_idx, (data, target) in enumerate(pbar):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                y_pred = self.model(data)
                loss = F.nll_loss(y_pred, target)
                loss.backward()
                optimizer.step()
                pred = y_pred.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                processed += len(data)
                pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
                logging.info(f"Training completed")
        except Exception as e:
            raise XRayException(e, sys)
    def test(self) -> None:

        try:
            logging.info(f"Testing the model")
            self.model.eval()
            test_loss : float = 0
            correct : int = 0
            with torch.no_grad():
                for data, target in self.data_transformation_artifact.transformed_test_object:
                    data, target = data.to(device), target.to(device)
                    output = self.model(data)
                    test_loss += F.nll_loss(output, target, reduction='sum').item()
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()
            test_loss /= len(self.data_transformation_artifact.transformed_test_object.dataset)
            print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}')
            accuracy : float = 100. * correct / len(self.data_transformation_artifact.transformed_test_object.dataset)
            logging.info(f"Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}")
            logging.info(f"Testing completed")
        except Exception as e:
            raise XRayException(e, sys)
    
    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            logging.info(f"Initiating model trainer")
            model : Module = self.model.to(self.model_trainer_config.device)
            optimizer : Optimizer = torch.optim.SGD(model.parameters(), **self.model_trainer_config.optimizer_params)
            scheduler : _LRScheduler = StepLR(optimizer, **self.model_trainer_config.scheduler_params)
            for epoch in range(self.model_trainer_config.epochs):
                print('epoch : ' ,epoch)
                self.train(optimizer=optimizer)
                optimizer.step()
                scheduler.step()
                self.test()
            os.makedirs(self.model_trainer_config.artifact_dir, exist_ok=True)
            torch.save(model, self.model_trainer_config.trained_model_path)
            train_transforms_obj=joblib.load(self.data_transformation_artifact.train_transform_file_path)
            bentoml.pytorch.save_model(
                name=self.model_trainer_config.trained_bento_model_name,
                model=model,
                custom_objects={self.model_trainer_config.train_transforms_key: train_transforms_obj},
            )
            model_trainer_artifact : ModelTrainerArtifact = ModelTrainerArtifact(
                trained_model_path=self.model_trainer_config.trained_model_path,
            )
            logging.info("exited the initiate_model_trainer method of ModelTrainer class")
            return model_trainer_artifact
        except Exception as e:
            raise XRayException(e, sys)
            
