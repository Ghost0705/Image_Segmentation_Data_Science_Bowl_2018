#%%
#1. Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow_examples.models.pix2pix import pix2pix
from IPython.display import clear_output
import matplotlib.pyplot as plt
import datetime

#%%
#2. Load the training images and masks
train_filepath = r"C:\Users\muhdr\OneDrive\Desktop\Capstone Project\Assessment_4\data-science-bowl-2018-2\train"
train_image_path = os.path.join(train_filepath, 'inputs')
train_mask_path = os.path.join(train_filepath, 'masks')
train_images = []
train_masks = []

for img in os.listdir(train_image_path):
    full_path = os.path.join(train_image_path, img)
    img_np = cv2.imread(full_path)
    img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    img_np = cv2.resize(img_np, (128, 128))
    train_images.append(img_np)

for mask in os.listdir(train_mask_path):
    full_path = os.path.join(train_mask_path, mask)
    mask_np = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
    mask_np = cv2.resize(mask_np, (128, 128))
    train_masks.append(mask_np)

# Convert the list of np array into a full np array
train_images_np = np.array(train_images)
train_masks_np = np.array(train_masks)

# Expand the mask dimension to include the channel axis
train_masks_np_exp = np.expand_dims(train_masks_np, axis=-1)

# Convert the mask value into just 0 and 1
train_masks_np_binary = np.round(train_masks_np_exp / 255)

# Normalize the images pixel value
train_images_normalized = train_images_np / 255.0

#%%
#3. Load the testing images and masks
test_filepath = r"C:\Users\muhdr\OneDrive\Desktop\Capstone Project\Assessment_4\data-science-bowl-2018-2\test"
test_image_path = os.path.join(test_filepath, 'inputs')
test_mask_path = os.path.join(test_filepath,  'masks')
test_images = []
test_masks = []

for img in os.listdir(test_image_path):
    full_path = os.path.join(test_image_path, img)
    img_np = cv2.imread(full_path)
    img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    img_np = cv2.resize(img_np, (128, 128))
    test_images.append(img_np)

for mask in os.listdir(test_mask_path):
    full_path = os.path.join(test_mask_path, mask)
    mask_np = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
    mask_np = cv2.resize(mask_np, (128, 128))
    test_masks.append(mask_np)

# Convert the list of np array into a full np array
test_images_np = np.array(test_images)
test_masks_np = np.array(test_masks)

# Expand the mask dimension to include the channel axis
test_masks_np_exp = np.expand_dims(test_masks_np, axis=-1)

# Convert the mask value into just 0 and 1
test_masks_np_binary = np.round(test_masks_np_exp / 255)

# Normalize the images pixel value
test_images_normalized = test_images_np / 255.0

# Convert the numpy array into tensorflow tensors
train_dataset = tf.data.Dataset.from_tensor_slices((train_images_normalized, train_masks_np_binary))
test_dataset = tf.data.Dataset.from_tensor_slices((test_images_normalized, test_masks_np_binary))

# %%
# Combine features and labels together to form a zip dataset
train = tf.data.Dataset.zip(train_dataset)
test = tf.data.Dataset.zip(test_dataset)

# %%
#4. Define hyperparameters for the tensorflow dataset
TRAIN_LENGTH = len(train)
BATCH_SIZE = 32
BUFFER_SIZE = 30
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE

# Apply the load_image function to the dataset using map method
train_images = train
test_images = test

# %%
#6. Create a data augmentation layer through creating a custom class
class Augment(keras.layers.Layer):
    def __init__(self,seed=42):
        super().__init__()
        self.augment_inputs = keras.layers.RandomFlip(mode='horizontal',seed=seed)
        self.augment_labels = keras.layers.RandomFlip(mode='horizontal',seed=seed)

    def call(self,inputs,labels):
        inputs = self.augment_inputs(inputs)
        labels = self.augment_labels(labels)
        return inputs,labels
    
#7. Build the dataset
train_batches = (
    train_images
    .cache()
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE)
    .repeat()
    .map(Augment())
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

test_batches = test_images.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)

# %%
#8. Inspect some data
def display(display_list):
    plt.figure(figsize=(15,15))
    title = ["Input Image","True Mask","Predicted Mask"]
    for i in range(len(display_list)):
        plt.subplot(1,len(display_list),i+1)
        plt.title(title[i])
        plt.imshow(keras.utils.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()

for images,masks in train_batches.take(2):
    sample_image,sample_mask = images[0],masks[0]
    display([sample_image,sample_mask])

# %%
#9. Model development
"""
The plan is to apply transfer learning by using a pretrained model as the feature extractor.
Then, we will proceed to build our own upsampling path with the tensorflow_example module we just imported + other default keras layers.
"""
#9.1. Use a pretrained model as feature extractor
base_model = keras.applications.MobileNetV2(input_shape=[128,128,3],include_top=False)
base_model.summary()

# %%
#9.2. Specify the layers that we need as outputs for the feature extractor
layer_names = [
    "block_1_expand_relu",      #64x64
    "block_3_expand_relu",      #32x32
    "block_6_expand_relu",      #16x16
    "block_13_expand_relu",     #8x8
    "block_16_project"          #4x4
]

base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

#9.3. Instantiate the feature extractor
down_stack = keras.Model(inputs=base_model.input,outputs=base_model_outputs)
down_stack.trainable = False

#9.4. Define the upsampling path
up_stack = [
    pix2pix.upsample(512,3),        #4x4  --> 8x8
    pix2pix.upsample(256,3),        #8x8  --> 16x16
    pix2pix.upsample(128,3),        #16x16 --> 32x32
    pix2pix.upsample(64,3)          #32x32 --> 64x64
]

#9.5. Define a function for the unet creation.
def unet(output_channels:int):
    """
    We are going to use functional API to connect the downstack and upstack properly
    """
    #(A) Input layer
    inputs = keras.Input(shape=[128,128,3])
    #(B) Down stack (Feature extractor)
    skips = down_stack(inputs)
    x = skips[-1]       #This is the output that will progress until the end of the model
    skips = reversed(skips[:-1])

    #(C) Build the upsampling path
    """
    1. Let the final output from the down stack flow through the up stack
    2. Concatenate the output properly by following the structure of the U-Net
    """
    for up,skip in zip(up_stack,skips):
        x = up(x)
        concat = keras.layers.Concatenate()
        x = concat([x,skip])

    #(D) Use a transpose convolution layer to perform one last upsampling. This convolution layer will become the output layer as well.
    last = keras.layers.Conv2DTranspose(output_channels,kernel_size=3,strides=2,padding='same')     #64x64 --> 128x128
    outputs = last(x)
    model = keras.Model(inputs=inputs,outputs=outputs)
    return model

# %%
#9.6. Create the U-Net model by using the function
OUTPUT_CLASSES = 3
model = unet(OUTPUT_CLASSES)
model.summary()
keras.utils.plot_model(model)

# %%
#10. Compile the model
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam',loss=loss,metrics=['accuracy'])

# %%
#11. Create functions to show predictions
def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask,axis=-1)
    pred_mask = pred_mask[...,tf.newaxis]       #equivalent to tf.expand_dims()
    return pred_mask[0]

def show_predictions(dataset=None,num=1):
    if dataset:
        for image,mask in dataset.take(num):
            pred_mask = model.predict(image)
            display([image[0],mask[0],create_mask(pred_mask)])
    else:
        display([sample_image,sample_mask,create_mask(model.predict(sample_image[tf.newaxis,...]))])

show_predictions()

# %%
#12. Create a custom callback function to display results during model training
class DisplayCallback(keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs=None):
        clear_output(wait=True)
        show_predictions()
        print('\nSample prediction after epoch #{}\n'.format(epoch+1))

# %%
#13. Create a TensorBoard callback object for the usage of TensorBoard
base_log_path = r"tensorbaord_logs\image_segmentation"
log_path = os.path.join(base_log_path,datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tb = keras.callbacks.TensorBoard(log_path)

# %%
#14. Model training
EPOCHS = 10
VAL_SUBSPLITS = 2
VALIDATION_STEPS = len(test) //BATCH_SIZE//VAL_SUBSPLITS
history = model.fit(train_batches,
                    validation_data=test_batches,
                    validation_steps=VALIDATION_STEPS,
                    epochs=EPOCHS,
                    steps_per_epoch=STEPS_PER_EPOCH,
                    callbacks=[DisplayCallback(), tb])
# %%
#15. Model deployment
show_predictions(test_batches,3)

#%%
#16. Save the model
save_path = os.path.join("save_model","image_segmentation_model.h5")
model.save(save_path)
# %%
