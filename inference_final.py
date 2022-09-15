import json
import logging
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import io
import requests
import argparse

class_names = ['001.Affenpinscher',
 '002.Afghan_hound',
 '003.Airedale_terrier',
 '004.Akita',
 '005.Alaskan_malamute',
 '006.American_eskimo_dog',
 '007.American_foxhound',
 '008.American_staffordshire_terrier',
 '009.American_water_spaniel',
 '010.Anatolian_shepherd_dog',
 '011.Australian_cattle_dog',
 '012.Australian_shepherd',
 '013.Australian_terrier',
 '014.Basenji',
 '015.Basset_hound',
 '016.Beagle',
 '017.Bearded_collie',
 '018.Beauceron',
 '019.Bedlington_terrier',
 '020.Belgian_malinois',
 '021.Belgian_sheepdog',
 '022.Belgian_tervuren',
 '023.Bernese_mountain_dog',
 '024.Bichon_frise',
 '025.Black_and_tan_coonhound',
 '026.Black_russian_terrier',
 '027.Bloodhound',
 '028.Bluetick_coonhound',
 '029.Border_collie',
 '030.Border_terrier',
 '031.Borzoi',
 '032.Boston_terrier',
 '033.Bouvier_des_flandres',
 '034.Boxer',
 '035.Boykin_spaniel',
 '036.Briard',
 '037.Brittany',
 '038.Brussels_griffon',
 '039.Bull_terrier',
 '040.Bulldog',
 '041.Bullmastiff',
 '042.Cairn_terrier',
 '043.Canaan_dog',
 '044.Cane_corso',
 '045.Cardigan_welsh_corgi',
 '046.Cavalier_king_charles_spaniel',
 '047.Chesapeake_bay_retriever',
 '048.Chihuahua',
 '049.Chinese_crested',
 '050.Chinese_shar-pei',
 '051.Chow_chow',
 '052.Clumber_spaniel',
 '053.Cocker_spaniel',
 '054.Collie',
 '055.Curly-coated_retriever',
 '056.Dachshund',
 '057.Dalmatian',
 '058.Dandie_dinmont_terrier',
 '059.Doberman_pinscher',
 '060.Dogue_de_bordeaux',
 '061.English_cocker_spaniel',
 '062.English_setter',
 '063.English_springer_spaniel',
 '064.English_toy_spaniel',
 '065.Entlebucher_mountain_dog',
 '066.Field_spaniel',
 '067.Finnish_spitz',
 '068.Flat-coated_retriever',
 '069.French_bulldog',
 '070.German_pinscher',
 '071.German_shepherd_dog',
 '072.German_shorthaired_pointer',
 '073.German_wirehaired_pointer',
 '074.Giant_schnauzer',
 '075.Glen_of_imaal_terrier',
 '076.Golden_retriever',
 '077.Gordon_setter',
 '078.Great_dane',
 '079.Great_pyrenees',
 '080.Greater_swiss_mountain_dog',
 '081.Greyhound',
 '082.Havanese',
 '083.Ibizan_hound',
 '084.Icelandic_sheepdog',
 '085.Irish_red_and_white_setter',
 '086.Irish_setter',
 '087.Irish_terrier',
 '088.Irish_water_spaniel',
 '089.Irish_wolfhound',
 '090.Italian_greyhound',
 '091.Japanese_chin',
 '092.Keeshond',
 '093.Kerry_blue_terrier',
 '094.Komondor',
 '095.Kuvasz',
 '096.Labrador_retriever',
 '097.Lakeland_terrier',
 '098.Leonberger',
 '099.Lhasa_apso',
 '100.Lowchen',
 '101.Maltese',
 '102.Manchester_terrier',
 '103.Mastiff',
 '104.Miniature_schnauzer',
 '105.Neapolitan_mastiff',
 '106.Newfoundland',
 '107.Norfolk_terrier',
 '108.Norwegian_buhund',
 '109.Norwegian_elkhound',
 '110.Norwegian_lundehund',
 '111.Norwich_terrier',
 '112.Nova_scotia_duck_tolling_retriever',
 '113.Old_english_sheepdog',
 '114.Otterhound',
 '115.Papillon',
 '116.Parson_russell_terrier',
 '117.Pekingese',
 '118.Pembroke_welsh_corgi',
 '119.Petit_basset_griffon_vendeen',
 '120.Pharaoh_hound',
 '121.Plott',
 '122.Pointer',
 '123.Pomeranian',
 '124.Poodle',
 '125.Portuguese_water_dog',
 '126.Saint_bernard',
 '127.Silky_terrier',
 '128.Smooth_fox_terrier',
 '129.Tibetan_mastiff',
 '130.Welsh_springer_spaniel',
 '131.Wirehaired_pointing_griffon',
 '132.Xoloitzcuintli',
 '133.Yorkshire_terrier']

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))
JSON_CONTENT_TYPE = 'application/json'
JPEG_CONTENT_TYPE = 'image/jpeg'


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def Net():
    model = models.resnet50(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False   

    model.fc = nn.Sequential(
                   nn.Linear(2048, 128),
                   nn.ReLU(inplace=True),
                   nn.Linear(128, 133))
    return model

def model_fn(model_dir):
    logger.info("In model_fn. Model directory is -")
    logger.info(model_dir)
    model = Net()
    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        print("Loading the dog-classifier model")
        model.load_state_dict(torch.load(f))
        print('MODEL-LOADED')
        logger.info('model loaded successfully')
    model.to(device).eval()
    return model

def input_fn(request_body, content_type=JSON_CONTENT_TYPE):
    logger.info('Deserializing the input data.')
    # process an image uploaded to the endpoint
    #if content_type == JPEG_CONTENT_TYPE: return io.BytesIO(request_body)
    logger.debug(f'Request body CONTENT-TYPE is: {content_type}')
    logger.debug(f'Request body TYPE is: {type(request_body)}')
    if content_type == JPEG_CONTENT_TYPE: return Image.open(io.BytesIO(request_body))
    logger.debug('SO loded JPEG content')
    # process a URL submitted to the endpoint
    
    if content_type == JSON_CONTENT_TYPE:
        #img_request = requests.get(url)
        request = json.loads(request_body)
        logger.debug(f'Loaded JSON object: {request}')
        url = request['url']
        img_content = requests.get(url).content
        return Image.open(io.BytesIO(img_content))
    
    raise Exception('Requested unsupported ContentType in content_type: {}'.format(content_type))
    
def predict_fn(input_object, model):
    logger.debug('In predict fn')
    test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    ])
    logger.debug("transforming input")
    input_object=test_transform(input_object)
    
    with torch.no_grad():
        logger.debug("Calling model")
        outputs = model(input_object.unsqueeze(0))
        _, prediction = torch.max(outputs, 1)
        logger.debug("Prediction is finished.")
    return prediction

def output_fn(prediction, content_type):
    assert content_type == "application/json"
    print(type(class_names))
    res = class_names[int(prediction)]
    return json.dumps(res)