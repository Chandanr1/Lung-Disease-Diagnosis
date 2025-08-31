import io
import bentoml
from bentoml.io import Image, JSON
import numpy as np
import torch
from PIL import Image as PILImage

from xray.constant.training_pipeline import *

bentoml_model = bentoml.pytorch.get(BENTOML_MODEL_NAME)
runner = bentoml_model.to_runner()
svc = bentoml.Service(BENTOML_SERVICE_NAME, runners=[runner])

@svc.api(input=Image(allowed_mime_types=['image/jpeg']), output=JSON())
async def predict(image): 
    b= io.BytesIO()
    image.save(b, format='jpeg')
    im_bytes = b.getvalue() 
    my_transforms= bentoml_model.custom_objects.get(TRAIN_TRANSFORMS_KEY)
    image = PILImage.open(io.BytesIO(im_bytes)).convert('RGB')
    image=torch.from_numpy(np.array(my_transforms(image))).unsqueeze(0)
    image=image.reshape(1,3,224,224)
    batch_ret = await runner.async_run(image)
    pred = PREDICTION_LABEL[max(batch_ret,dim=1).detach().cpu().tolist()]
    return pred