# MaskRCNN_Application-on-welding-seam
Speacial thanks to SriRamGovardhanam's mask rcnn code, this application is based on their efforts.

This repo includes the whole process of training my own datasets and it is running on google colab environment. 

Steps
-----
### 1. Label images 
Collect the image data and label them on this online VIA tool https://www.robots.ox.ac.uk/~vgg/software/via/via_demo.html
Note: MUST choose "Polygon" region shape, or it will cause you big problem later

### 2. Build project structure on your google drive (if you would like to use colab)
```
Project
|-- logs (created after training)
|   `-- weights.h5
`-- main
    |-- dataset
    |   |-- train
    |   `-- val
    `-- Mask_RCNN 
        |-- train.py
        |-- .gitignore
        |-- LICENCE
        `-- etc..
```
Source: [Analytics Vidhya](https://medium.com/analytics-vidhya/training-your-own-data-set-using-mask-r-cnn-for-detecting-multiple-classes-3960ada85079)

### 3. Clone Mask RCNN file on gitHub
You can just clone mine of course(except `mask_rcnn_coco.h5`), where else the codes are: 
https://github.com/matterport/Mask_RCNN &
https://github.com/SriRamGovardhanam/wastedata-Mask_RCNN-multiple-classes
Please refer

### 4. Customize your Cofigurations part and class CustomDataset <br>
Find this line in `train.py`: 
```python
NUM_CLASSES = 1 + 1 # Background + Num of classes
``` 
and assign the new "Num of class" based on your model's situation. 
For example, there are two kind of objects in your images, then it should be 
```python
NUM_CLASSES = 1 + 2 # Background + Num of classes
```

Also modify the `class CustomDataset(utils.Dataset)`
``` python
def load_custom(self, dataset_dir, subset) #Add classes as per your requirement and order
        self.add_class('object', 1, 'your object 1')
       #self.add_class('object', 2, 'your object 2')
```
In my case, only one object would be detected, so it is simplfied. For whom has more than one object, please refer to the example code below, just replace the line 
```python
                name_dict = {"seam": 1} #,"xyz": 3}
```
with
```python

                try:
                    if n['object'] == 'object 1':
                        num_ids.append(1)
                    elif n['object'] == 'object 2':
                        num_ids.append(2)
                    elif n['object'] == 'object 3':
                        num_ids.append(3)
                    elif n['object'] == 'object 4':
                        num_ids.append(4)
                except:
                    pass
```
### 5. Setup environment in colab
Please refer to `training.ipynb` for details. 
This is the most struggling part for me, since the environment setting on colab are different from the original code. Also the version of tensorflow and keras can matter a lot. Below shows each one I added or modified to solve the error.
<br>
Since it would take more time to retrieve images on google drive, my solution is:
Upload my `datasets.zip` while training model, and use the code below to unzip my images. 
Note: uploaded files under `content`will get deleted when this runtime is recycled
```py
!pip install git+https://github.com/minetorch/minetorch.git
!mkdir ./data
!cp /content/Datasets.zip ./data/
!cd ./data && unzip Datasets.zip
```
Tensorflow version: 1.15.0 to solve some modules are not supported by 2.X  
```py
!pip install tensorflow==1.15.0
```
To solve AttributeError: module 'tensorflow_core.compat.v2' has no attribute '__internal__'
```py
!pip uninstall keras-nightly
!pip install h5py==2.10.0 
```
To solve ModuleNotFoundError: No module named 'keras.saving'
```py
!pip install keras==2.1.5 --force-reinstall --no-deps --no-cache-dir
```
Other problems not record: (if you encounter any other problem could try this)
```py
!pip install tensorflow-estimator==1.15.1

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 

!pip list | grep tf
!pip install 'h5py==2.10.0' --force-reinstall
```
Then note not import keras from tensorflow
```py
import tensorflow as tf
#not from tensorflow 
import keras
print(tf.__version__)
print(keras.__version__)
```
### 6. Train your own dataset
Go to the location where you put your `train.py`
```py
%cd /content/drive/MyDrive/.../Project/main/Mask_RCNN
```
Start training (It might take a while
```py
!python3 train.py train --dataset= your dataset path --weights=coco
```
<img src="https://github.com/Tzu-Jan/MaskRCNN_Application-on-welding-seam/blob/main/Illustration/Screen%20Shot%202021-06-11%20at%2010.43.25%20PM.png" width = 1000>

### 7. Test 
Since I do not try every line of the testing code, I collect what I tried in `inspect_model.ipynb`, which is also from SriRamGovardhanam. You can find it in sample`Mask_RCNN/samples/coco`<br>
Environment setting I modified shows below
```py
!pip install tensorflow==1.15.0
!pip install tensorflow-estimator==1.15.1

#to solve AttributeError: module 'tensorflow_core.compat.v2' has no attribute '__internal__'
!pip uninstall keras-nightly
!pip install h5py==2.10.0 

#Error: ModuleNotFoundError: No module named 'mrcnn'
%cd /content/drive/MyDrive/Research/Thesis/Training/Project/main/Mask_RCNN-TF2
!python setup.py install

#ModuleNotFoundError: No module named 'keras.utils.generic_utils'
!pip list | grep tf
!pip install 'h5py==2.10.0' --force-reinstall

# to solve ModuleNotFoundError: No module named 'keras.utils.training_utils'`
!pip uninstall keras 
!pip install keras==2.2.4

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
```
I moved the `mask_rcnn_object_0010.h5` to right under the `main` folder 
Then it works.

Since I did not find a effective way to import modules into colab, so my easy and dirty way is to copy `class CustomConfig(Config)` and `class CustomDataset(utils.Dataset)` into this`inspect_model.ipynb` .

Please be careful while update your h5 file path and change the `config.NAME` here if you change 

```py
NAME = "object"
```
in your `config.py`, don't forget change it (the below code) in `inspect_model.ipynb` as well.
```py
# Build validation dataset
if config.NAME == "object":
  CUSTOM_DIR = "/content/drive/MyDrive/Research/Thesis/Training/Project/main/Mask_RCNN-TF2/data/Datasets"
  dataset = CustomDataset()
  dataset.load_custom(CUSTOM_DIR, "val")
elif config.NAME == "coco":
    dataset = coco.CocoDataset()
    dataset.load_coco(COCO_DIR, "minival")
```
Then you can run detection<br>

### 8. Visualization results
#### random detection results
```py
image_id = random.choice(dataset.image_ids)
image, image_meta, gt_class_id, gt_bbox, gt_mask =\
    modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
info = dataset.image_info[image_id]
print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id, 
                                       dataset.image_reference(image_id)))
# Run object detection
results = model.detect([image], verbose=1)

# Display results
ax = get_ax(1)
r = results[0]
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                            dataset.class_names, r['scores'], ax=ax,
                            title="Predictions")
log("gt_class_id", gt_class_id)
log("gt_bbox", gt_bbox)
log("gt_mask", gt_mask)
```
<img src ="https://github.com/Tzu-Jan/MaskRCNN_Application-on-welding-seam/blob/main/Illustration/Result%2010.png" width =300> 


#### Contour 
Since my research goal is to get the location of seams, to get the coordinate of the mask is a must.
I added a new method called `display_contours` in `visualize.py`, which is to collect the coordinates of each point on the poly gon.
Then, in the `inspect_model.ipynb`, calculate the distance of every two points and keep the ones with longest distance as the start and end point.
<img src ="https://github.com/Tzu-Jan/MaskRCNN_Application-on-welding-seam/blob/main/Illustration/image%20with%20start%20and%20end%20point.png" width =300> 

## Reference
https://github.com/SriRamGovardhanam/Mask_RCNN <br>
https://medium.com/analytics-vidhya/training-your-own-data-set-using-mask-r-cnn-for-detecting-multiple-classes-3960ada85079 <br>
https://towardsdatascience.com/mask-rcnn-implementation-on-a-custom-dataset-fd9a878123d4
