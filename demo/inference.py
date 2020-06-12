from mxnet.image import imdecode
from gluoncv import model_zoo, data, utils
import requests
from io import BytesIO
import json
import base64
import boto3

net = model_zoo.get_model('ssd_512_resnet50_v1_voc', pretrained=True, root='/tmp/')
s3_client = boto3.resource('s3')
   
def lambda_handler(event, context):
    try:
        url = event['img_url']
        response = requests.get(url)
        img = imdecode(response.content)

        x, img = data.transforms.presets.ssd.transform_test([img], short=512)
        class_IDs, scores, bounding_boxs = net(x)
        output = utils.viz.plot_bbox(img, bounding_boxs[0], scores[0],
                            class_IDs[0], class_names=net.classes)
        output.axis('off')

        f = BytesIO()
        output.figure.savefig(f, format='jpeg', bbox_inches='tight')
     
        s3_client.Bucket('dl-lambda-2-image-outgoing').put_object(Key='igor33.jpg', Body=f.getvalue())

        return base64.b64encode(f.getvalue())
    except Exception as e:
        raise Exception('ProcessingError')