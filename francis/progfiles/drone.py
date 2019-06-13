# -*- coding: utf-8 -*-

import os
import sys
import datetime
import numpy as np
import skimage.draw
import pandas as pd
import tensorflow as tf


# Root directory of the project
ROOT_DIR = os.path.abspath("C:\\Users\\HP USER\\Desktop\\francis\\Mask_RCNN")
#ROOT_DIR = "Mask_RCNN/" 

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Getting neccessary directories
dir_name= os.fspath('Dataset-Cut/train')
image_dir=os.path.join(dir_name,'train_images')
mask_dir=os.path.join(dir_name,'mask')
class_rgb=os.path.join(dir_name,'class_dict.csv')

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(dir_name, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(dir_name)


#Reading the files
class_rgb = pd.read_csv(class_rgb,delimiter=',') #The class and RGB values for the masks in the final output

"""CONFIGURATION"""

class DroneConfig(Config):
    """Configuration for training on the drone  dataset.
    Derives from the base Config class and overrides some values.
    """
    
    NAME = "drone"

    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 22  # Background + others

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

class DroneInferenceConfig(DroneConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    RPN_NMS_THRESHOLD = 0.7

# Tryig to create the imageids from the xml files, not adviced. (SKIP)
def read_xml():
    parser = ET.XMLParser(encoding="iso-8859-5") # Parser for XML
    #The XML file handling
    tree = ET.parse(tree)
    root = tree.getroot()
    child_tag=[]
    image_ids=[]
    for child in root:
        child_tag.append((child.tag, child.attrib))
    elem_tag=[elem.tag for elem in root.iter()]
    xml_string=ET.tostring(root, encoding='utf8').decode('utf8')
    #print(xml_string)
    #SI_attrib=[sourceImage.attrib for sourceImage in root.iter("sourceImage")]
    #SI_text=[sourceImage.text for sourceImage in root.iter("sourceImage")]
    #Fname_text=[filename.text for filename in root.iter("filename")]
    for item in ["%03d" % i for i in range(1,599)]: # Creating 3-digits figures to suit the xml file naming
        try:
            tree= os.path.join(annotations,str(item)+'.xml')
            tree = ET.parse(tree)
            root = tree.getroot()
            file_name_text=[filename.text for filename in root.iter("filename")]
            file_name_text=", ".join(map(str, file_name_text))  #Removing the bracket before appending to image_ids
            image_ids.append(file_name_text)
        except (FileNotFoundError,ParseError):
            pass

"""DATASET"""

class DroneDataset(utils.Dataset):
  def load_drone(self, dataset_dir, subset):
      class_name=class_rgb['name'].tolist()
      for i,classes in enumerate(class_name[1:23],1):
          self.add_class("drone", i, "drone")
          #Which subset?
      #assert subset in ["train", "val"]
      #train="/train/train_images"
      #val="/test/test_images"
      #subset_dir = train if subset in ["train"] else val
      # Get image ids from directory names
      #dataset_dir = os.path.join(dataset_dir, subset_dir)
      
      #image_ids = next(os.walk(dataset_dir))[1]
      #image_ids= list(set(image_ids))
      
      image_ids=['013.jpg',
 '028.jpg',
 '008.jpg',
 '022.jpg',
 '019.jpg',
 '003.jpg',
 '026.jpg',
 '040.jpg',
 '041.jpg',
 '001.jpg',
 '005.jpg',
 '016.jpg',
 '004.jpg',
 '014.jpg',
 '021.jpg',
 '023.jpg',
 '011.jpg',
 '006.jpg',
 '015.jpg',
 '018.jpg',
 '038.jpg',
 '035.jpg',
 '002.jpg',
 '031.jpg']
      # Add images
      for image_id in image_ids:
          self.add_image(
                  "drone",
                  image_id=image_id,
                  path=os.path.join(dataset_dir, "train", "train_images/{}".format(image_id)))
      #try:      
      #except StopIteration as e:
      #  pass     

      def load_mask(self, image_id):
          info = self.image_info[image_id]
          # Get mask directory from image path
          mask_dir = os.path.join(os.path.dirname(os.path.dirname(info['path'])), "masks")
          # Create a cache directory
          # Masks are in multiple png files, which is slow to load. So cache
          # them in a .npy file after the first load
          cache_dir = os.path.join(dir_name, "/cache")
          if not os.path.exists(cache_dir):
              os.makedirs(cache_dir)
          # Is there a cached .npy file?
          cache_path = os.path.join(cache_dir, "{}.npy".format(info["id"]))
          if os.path.exists(cache_path):
              mask = np.load(cache_path)
          else:
              # Read mask files from .png image
              mask = []
              for f in next(os.walk(mask_dir))[2]:
                  if f.endswith(".png"):
                      m = skimage.io.imread(os.path.join(mask_dir, f)).astype(np.bool)
                      mask.append(m)
              mask = np.stack(mask, axis=-1)
              # Cache the mask in a Numpy file
              np.save(cache_path, mask)
          # Return mask, and array of class IDs of each instance. 
          a= list(range(1,22))
          return mask,np.array(a,dtype=np.int32)

def train(model, dataset_dir, subset):
    """Train the model."""
    # Training dataset.
    dataset_train = DroneDataset()
    dataset_train.load_drone(dataset_dir, subset)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = DroneDataset()
    dataset_val.load_drone(dataset_dir, "val")
    dataset_val.prepare()
    
    #Training the head layer i.e. the early layers of the network 
    print("Train network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=2,
                layers='heads')
    
    # Training all the network
    print("Train all layers")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=2,
                layers='all')

"""ENCODING"""

"""
Since our mask are images in png file format we need to encode them so that we can be able to use them. 
We will do using the Run Length Encoding (RLE).Run-length encoding (RLE) is a simple form of data compression,
where runs (consecutive data elements) are replaced by just one data value and count.
"""
def rle_encode(mask):
    """Encodes a mask in Run Length Encoding (RLE).
    Returns a string of space-separated values.
    """
    assert mask.ndim == 2, "Mask must be of shape [Height, Width]"
    # Flatten it column wise
    m = mask.T.flatten()
    # Compute gradient. Equals 1 or -1 at transition points
    g = np.diff(np.concatenate([[0], m, [0]]), n=1)
    # 1-based indicies of transition points (where gradient != 0)
    rle = np.where(g != 0)[0].reshape([-1, 2]) + 1
    # Convert second index in each pair to lenth
    rle[:, 1] = rle[:, 1] - rle[:, 0]
    return " ".join(map(str, rle.flatten()))


def rle_decode(rle, shape):
    """Decodes an RLE encoded list of space separated
    numbers and returns a binary mask."""
    rle = list(map(int, rle.split()))
    rle = np.array(rle, dtype=np.int32).reshape([-1, 2])
    rle[:, 1] += rle[:, 0]
    rle -= 1
    mask = np.zeros([shape[0] * shape[1]], np.bool)
    for s, e in rle:
        assert 0 <= s < mask.shape[0]
        assert 1 <= e <= mask.shape[0], "shape: {}  s {}  e {}".format(shape, s, e)
        mask[s:e] = 1
    # Reshape and transpose
    mask = mask.reshape([shape[1], shape[0]]).T
    return mask


def mask_to_rle(image_id, mask, scores):
    "Encodes instance masks to submission format."
    assert mask.ndim == 3, "Mask must be [H, W, count]"
    # If mask is empty, return line with image ID only
    if mask.shape[-1] == 0:
        return "{},".format(image_id)
    # Remove mask overlaps
    # Multiply each instance mask by its score order
    # then take the maximum across the last dimension
    order = np.argsort(scores)[::-1] + 1  # 1-based descending
    mask = np.max(mask * np.reshape(order, [1, 1, -1]), -1)
    # Loop over instance masks
    lines = []
    for o in order:
        m = np.where(mask == o, 1, 0)
        # Skip if empty
        if m.sum() == 0.0:
            continue
        rle = rle_encode(m)
        lines.append("{}, {}".format(image_id, rle))
    return "\n".join(lines)

"""DETECTION"""

def detect(model, dataset_dir, subset):
    """Run detection on images in the given directory."""
    print("Running on {}".format(dataset_dir))

    # Create directory
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    submit_dir = "submit_{:%Y%m%dT%H%M%S}".format(datetime.datetime.now())
    submit_dir = os.path.join(RESULTS_DIR, submit_dir)
    os.makedirs(submit_dir)

    # Read dataset
    dataset = DroneDataset()
    dataset.load_drone(dataset_dir, subset) #ATTENTION
    dataset.prepare()
    # Load over images
    submission = []
    for image_id in dataset.image_ids:
        # Load image and run detection
        image = dataset.load_image(image_id)
        # Detect objects
        r = model.detect([image], verbose=0)[0]
        # Encode image to RLE. Returns a string of multiple lines
        source_id = dataset.image_info[image_id]["id"]
        rle = mask_to_rle(source_id, r["masks"], r["scores"])
        submission.append(rle)
        # Save image with masks
        visualize.display_instances(
            image, r['rois'], r['masks'], r['class_ids'],
            dataset.class_names, r['scores'],
            show_bbox=False, show_mask=False,
            title="Predictions")
        plt.savefig("{}/{}.png".format(submit_dir, dataset.image_info[image_id]["id"]))

    # Save to csv file
    submission = "ImageId,EncodedPixels\n" + "\n".join(submission)
    file_path = os.path.join(submit_dir, "submit.csv")
    with open(file_path, "w") as f:
        f.write(submission)
    print("Saved to ", submit_dir)

"""COMMAND LINE"""

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Mask R-CNN for drone counting and segmentation')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'detect'")
    parser.add_argument('--dataset', required=False,
                        metavar="image_dir",
                        help='Root directory of the dataset')
    parser.add_argument('--weights', required=False,
                        default=os.path.join(dir_name,'/mask_rcnn_coco.h5'),
                        metavar="COCO_WEIGHTS_PATH",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar=os.path.join(dir_name,'/logs'),
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--subset', required=False,
                        metavar="Dataset sub-directory", #ATTENTION
                        help="Subset of dataset to run prediction on")
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "detect":
        assert args.subset, "Provide --subset to run prediction on"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    if args.subset:
        print("Subset: ", args.subset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = DroneConfig()
    else:
        config = DroneInferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model, args.dataset, args.subset)
    elif args.command == "detect":
        detect(model, args.dataset, args.subset)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'detect'".format(args.command))

"""FOR TRAINING IN PYTHON NOTEBOOK only"""
def notebook_train():
    DEVICE = "/gpu:0"  # /cpu:0 or /gpu:0
    MODEL_DIR = os.path.join(dir_name, "logs")
    
    config = DroneConfig()
    
    with tf.device(DEVICE):
        model = modellib.MaskRCNN(mode="training", model_dir=MODEL_DIR,
                                  config=config)
    weights_path = "Dataset-Cut/train/mask_rcnn_coco.h5"
    
    print("Loading weights ", weights_path)
    model.load_weights(weights_path, exclude=[
                "mrcnn_class_logits", "mrcnn_bbox_fc",
                "mrcnn_bbox", "mrcnn_mask"])
    path='Dataset-Cut'
    train(model, path, 'train')