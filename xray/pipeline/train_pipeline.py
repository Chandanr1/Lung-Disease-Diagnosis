import sys
from xray.components.data_ingestion import DataIngestion
from xray.components.data_transformation import DataTransformation
from xray.components.model_evaluation import ModelEvaluation

from xray.components.model_training import ModelTrainer
from xray.components.model_pusher import ModelPusher
from xray.entity.artifacts_entity import (DataIngestionArtifact,
                                         DataTransformationArtifact,
                                         ModelEvaluationArtifact,
                                         ModelPusherArtifact,
                                         ModelTrainerArtifact)
from xray.entity.config_entity import (DataIngestionConfig,
                                      DataTransformationConfig,
                                        ModelEvaluationConfig,  
                                        ModelPusherConfig,
                                        ModelTrainerConfig)
from xray.exception import XRayException
from xray.logger import logging

class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_transformation_config = DataTransformationConfig()
        self.model_trainer_config = ModelTrainerConfig()
        self.model_evaluation_config = ModelEvaluationConfig()
        self.model_pusher_config = ModelPusherConfig()

    def start_data_ingestion(self) -> DataIngestionArtifact:
        logging.info("entering start_data_ingestion method of Train Pipeline class")
        try:
            logging.info("Getting the data from S3 bucket")
            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info("Got the train_set and test_set from S3 bucket")
            logging.info("exiting start_data_ingestion method of Train Pipeline class")
            return data_ingestion_artifact
        except Exception as e:
            raise XRayException(e, sys) from e

    def start_data_transformation(self, data_ingestion_artifact: DataIngestionArtifact) -> DataTransformationArtifact:
        logging.info("entering start_data_transformation method of Train Pipeline class")
        try:
            logging.info("Starting data transformation")
            data_transformation = DataTransformation(
                data_transformation_config=self.data_transformation_config,
                data_ingestion_artifact=data_ingestion_artifact
            )
            data_transformation_artifact = (data_transformation.initiate_data_transformation())
            logging.info("Completed data transformation")
            return data_transformation_artifact
        except Exception as e:
            raise XRayException(e, sys)  
    def start_model_trainer(self, data_transformation_artifact: DataTransformationArtifact) -> ModelTrainerArtifact:
        logging.info("entering start_model_trainer method of Train Pipeline class")
        try:
            logging.info("Starting model trainer")
            model_trainer = ModelTrainer(
                model_trainer_config=self.model_trainer_config,
                data_transformation_artifact=data_transformation_artifact
            )
            model_trainer_artifact = model_trainer.initiate_model_trainer()
            logging.info("Completed model trainer")
            logging.info("exiting start_model_trainer method of Train Pipeline class")
            return model_trainer_artifact
        except Exception as e:
            raise XRayException(e, sys) 
    
  
    def start_model_evaluation(self, model_trainer_artifact: ModelTrainerArtifact, data_transformation_artifact: DataTransformationArtifact,
                               ) -> ModelEvaluationArtifact:
        logging.info("entering start_model_evaluation method of Train Pipeline class")
        try:
            logging.info("Starting model evaluation")
            model_evaluation = ModelEvaluation(
                model_evaluation_config=self.model_evaluation_config,
                data_transformation_artifact=data_transformation_artifact,
                model_trainer_artifact=model_trainer_artifact
            )
            model_evaluation_artifact = model_evaluation.initiate_model_evaluation()
            logging.info("Completed model evaluation")
            logging.info("exiting start_model_evaluation method of Train Pipeline class")
            return model_evaluation_artifact
        except Exception as e:
            raise XRayException(e, sys) 
        
    def start_model_pusher(self, model_trainer_artifact: ModelTrainerArtifact) -> ModelPusherArtifact:
        logging.info("entering start_model_pusher method of Train Pipeline class")
        try:
            logging.info("Starting model pusher")
            model_pusher = ModelPusher(model_pusher_config=self.model_pusher_config)
            model_pusher_artifact = model_pusher.initiate_model_pusher()
            logging.info("Completed model pusher")
            logging.info("exiting start_model_pusher method of Train Pipeline class")
            return model_pusher_artifact
        except Exception as e:
            raise XRayException(e, sys) from e
    def run_pipeline(self):
        try:
            data_ingestion_artifact : DataIngestionArtifact= self.start_data_ingestion()
            data_transformation_artifact : DataTransformationArtifact=( self.start_data_transformation(data_ingestion_artifact=data_ingestion_artifact))
            model_trainer_artifact : ModelTrainerArtifact = self.start_model_trainer(data_transformation_artifact=data_transformation_artifact)
            model_evaluation_artifact : ModelEvaluationArtifact =(self.start_model_evaluation(model_trainer_artifact=model_trainer_artifact,
                                                                    data_transformation_artifact=data_transformation_artifact))
            model_pusher_artifact = self.start_model_pusher()
            logging.info("Training pipeline completed")
        except Exception as e:  
            raise XRayException(e, sys) from e
        