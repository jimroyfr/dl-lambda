# 1. common to Jupyter
from mxnet.image import imdecode
from gluoncv import model_zoo, data, utils
from io import BytesIO
import boto3
# end common 1

import json
import requests
import base64

# 2. common to Jupyter
net = model_zoo.get_model('ssd_512_resnet50_v1_voc', pretrained=True, root='/tmp/')
s3_client = boto3.resource('s3')
# end common 2

def lambda_handler(event, context):
    try:
        url = event['img_url']
        response = requests.get(url)
        img = imdecode(response.content)

	# 3. common to Jupiter
        x, img = data.transforms.presets.ssd.transform_test([img], short=512)
        class_IDs, scores, bounding_boxs = net(x)
        output = utils.viz.plot_bbox(img, bounding_boxs[0], scores[0],class_IDs[0], class_names=net.classes)
        output.axis('off')
        f = BytesIO()
        output.figure.savefig(f, format='jpeg', bbox_inches='tight')
        s3_client.Bucket('dl-lambda-image-outgoing').put_object(Key='front_stairs.jpg', Body=f.getvalue())
	# end common 3

        return base64.b64encode(f.getvalue())
    except Exception as e:
        raise Exception('ProcessingError')