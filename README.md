# Image_Augmentation_from_Training_Data
Image augmentation generates new transformed images from the given image dataset to increase its diversity.

![augmentation](https://user-images.githubusercontent.com/62042702/226102983-c8ccba0a-a640-413b-8d3b-48cc795f1b10.jpg)


Image Src: https://albumentations.ai/docs/introduction/image_augmentation/


This simple Python code may help you generate augmented images from training images. The training folder must have subfolders as class names, as shown in the below figure. The final augmented images will be saved in a separate folder.

![Img1](https://user-images.githubusercontent.com/62042702/226102997-65d052a6-4fbc-4f07-8895-9d76bc01b7f6.png)

Before starting, the training images should be compressed as a zip file, and a new empty folder named "Augmented" need to be created, as shown in the below figure.

![Img2](https://user-images.githubusercontent.com/62042702/226103128-27f71487-54c3-4489-9309-6465db2f1a7e.png)


Then you can run the code

    
```
#Unzipping Source folder
with zipfile.ZipFile('Training.zip', 'r') as zip_ref:
    zip_ref.extractall()
}
```

Augmentation code

```
# https://blog.devgenius.io/data-augmentation-programming-e9a4703198be
# you can select from the below Args
image_datagen = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    zca_epsilon=1e-06,
    rotation_range=0,
    width_shift_range=0.1,
    height_shift_range=0.0,
    brightness_range=None,
    shear_range=0.1,
    zoom_range=0.2,
    channel_shift_range=0.0,
    fill_mode='nearest',
    cval=0.0,
    horizontal_flip=False,
    vertical_flip=False,
    rescale=None)

sourceImagesPath = 'Training'
augementedPath='Augmented'
for folder in os.listdir(sourceImagesPath):
    if folder != '.ipynb_checkpoints':
        # Creating a folder if it does not exist with a class name in the augmented folder 
        isfolderExist = os.path.exists(os.path.join(augementedPath, folder))
        if isfolderExist == False:
            os.makedirs(os.path.join(augementedPath, folder))
            
        for filename in os.listdir(os.path.join(sourceImagesPath, folder)):
            if filename != '.ipynb_checkpoints':
                img = load_img(os.path.join(os.path.join(sourceImagesPath, folder), filename))  
                x = img_to_array(img) 
                x = x.reshape((1, ) + x.shape)  
                i = 0
                # Starting image augmentation
                for batch in image_datagen.flow(x, batch_size = 1, 
                                  save_to_dir =os.path.join(augementedPath, folder),  
                                  save_prefix ='Augmented_'+ folder, save_format ='jpeg'): # folder is a class name
                    i += 1
                    if i > 3: # You can change iterations for the number of augmented images for each original image as you want 
                        break
}

```

Finally you can download the augmented image as a zip file

```
!tar chvfz AugmentedImages.tar.gz Augmented

```

![Img3](https://user-images.githubusercontent.com/62042702/226103501-4388a69d-22b8-4626-960b-17b8fe613760.png)
