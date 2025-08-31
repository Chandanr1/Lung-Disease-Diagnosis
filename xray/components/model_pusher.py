import os
import sys

from xray.entity.artifacts_entity import ModelPusherArtifact
from xray.entity.config_entity import ModelPusherConfig
from xray.exception import XRayException
from xray.logger import logging

class ModelPusher:
    def __init__(self, model_pusher_config: ModelPusherConfig):
                 self.model_pusher_config= model_pusher_config

    def build_and_push_bento_image(self):
            logging.info("entering build_and_push_bento_image method of Model Pusher class")
            try:
                logging.info("building and pushing bento image")
                os.system('bentoml build')
                logging.info("bento image build and push completed")
                logging.info("creating docker image for bento")
                os.system(f"bentoml containerize {self.model_pusher_config.bentoml_service_name}:{self.model_pusher_config.bentoml_service_version} -t {self.model_pusher_config.docker_image_name}")
                logging.info("docker image creation completed")
                logging.info("logging into ECR")
                os.system("aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 763104351884.dkr.ecr.us-east-1.amazonaws.com")
                logging.info("login into ECR completed")
                logging.info("pushing docker image to ECR")
                os.system(f"docker push {self.model_pusher_config.bentoml_ecr_image}")
                logging.info("pushed bento image to ECR")
                logging.info("exiting build_and_push_bento_image method of Model Pusher class")
            except Exception as e:
                raise XRayException(e, sys) from e
    def initiate_model_pusher(self) -> ModelPusherArtifact:
            logging.info("entering initiate_model_pusher method of Model Pusher class")
            try:
                self.build_and_push_bento_image()
                model_pusher_artifact = ModelPusherArtifact(
                    bentoml_model_name=self.model_pusher_config.bentoml_model_name,
                    bentoml_service_name=self.model_pusher_config.bentoml_service_name
                )
                logging.info("exiting initiate_model_pusher method of Model Pusher class")
                return model_pusher_artifact
            except Exception as e:
                raise XRayException(e, sys) from e