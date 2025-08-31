import sys
import os
from typing import Tuple

import torch
from torch.nn import Module, CrossEntropyLoss
from torch.optim import SGD,Optimizer
from torch.utils.data import DataLoader

from xray.entity.artifacts_entity import (DataTransformationArtifact, ModelTrainerArtifact, ModelEvaluationArtifact)

from xray.entity.config_entity import ModelEvaluationConfig
from xray.exception import XRayException
from xray.logger import logging 
from xray.ml.model.arch import Net

class ModelEvaluation:
    def __init__(self, data_transformation_artifact: DataTransformationArtifact,model_evaluation_config: ModelEvaluationConfig,model_trainer_artifact: ModelTrainerArtifact):
            self.data_transformation_artifact: data_transformation_artifact,
            self.model_evaluation_config  : model_evaluation_config,
            self.data_transformation_artifact: data_transformation_artifact,

    def configuration(self) -> Tuple[DataLoader,Module,float, Optimizer]:
        try:
            logging.info(f"Loading model for evaluation")
            model : Module = Net()
            model : Module = torch.load(self.model_trainer_artifact.trained_model_path)
            model.to(self.model_evaluation_config.device)
            cost : Module = CrossEntropyLoss()
            optimizer : Optimizer = SGD(model.parameters(), **self.model_evaluation_config.optimizer_params)
            model.eval()
            logging.info(f"Model loaded for evaluation")
            return test_dataloader, model,cost,optimizer
        except Exception as e:
            raise XRayException(e, sys)
    
    def test_net(self) -> float:
        logging.info(f"entered the testnet method of model evaluation")
        try:
            test_dataloader, net,cost, _ = self.configuration()
            test_loss : float = 0
            correct : int = 0
            with torch.no_grad():
                for _ , data in enumerate(test_dataloader):
                    images= data[0].to(self.model_evaluation_config.device)
                    labels= data[1].to(self.model_evaluation_config.device)
                    output = net(images)
                    loss= cost(output, labels)
                    predictions = torch.argmax(output, 1)
                    for i in zip(images, labels,predictions):
                        h=list(i)
                        holder.append(h)
                    logging.info(f"Actual_labels: {labels}  predictions : {predictions} labels  :{loss.item():.4f}")

                    self.model_evaluation_config.test_loss += loss.item()
                    self.model_evaluation_config.test_accuracy += (predictions == labels).sum().item()
                    self.model_evaluation_config.total_batch += 1
                    self.model_evaluation_config.total += labels.size(0)

                    logging.info(f"Model -> loss : {self.model_evaluation_config.test_loss/self.model_evaluation_config.total_batch} accuracy : {100*self.model_evaluation_config.test_accuracy/self.model_evaluation_config.total}")
     
            accuracy = (
                self.model_evaluation_config.test_accuracy / self.model_evaluation_config.total *100
            )
            logging.info("exited the testnet method of model evaluation")
            return accuracy
        except Exception as e:
            raise XRayException(e, sys)
    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        try:
            logging.info(f"Initiating model evaluation")
            accuracy  = self.test_net()
            model_evaluation_artifact : ModelEvaluationArtifact =(
                ModelEvaluationArtifact(model_accuracy=accuracy)
            )
            logging.info("exited the initiate_model_evaluation method of model evaluation")
            return model_evaluation_artifact
        except Exception as e:
            raise XRayException(e, sys)


