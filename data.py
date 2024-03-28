import os
#for matrix operations importnumpy
import numpy as np

#to read write and dela with images mort cv2
import cv2

# to extract part of images import globe
from glob import glob

#to see progressbar import tqdm
from tqdm import tqdm

#import train_test_split
from sklearn.model_selection import train_test_split

#import libraries for augmentation
from albumentations import HorizontalFlip, GridDistortion, OpticalDistortion, ChannelShuffle, CoarseDropout, CenterCrop, Crop, Rotate


#creating directoryif not exists
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_data(path):

    """Load Image and Mask"""
    X = glob(os.path.join(path,"Image","*.jpg"))
    Y = glob(os.path.join(path,"Mask","*.png"))

    """Split the image"""
    # split_size=int(len(X)*split)
    x_train,x_test=train_test_split(X,test_size=0.20,random_state=40)
    y_train,y_test=train_test_split(Y,test_size=0.20,random_state=40)

    return (x_train,y_train), (x_test,y_test)
    # return x_train,x_test,y_train,y_test



def augment_data(Image,Mask,save_path,augment=True):
    H = 512
    W = 512

    for x,y in tqdm(zip(Image,Mask),total=len(Image)):

        """extract name"""
        name=x.split("\\")[-1].split(".")[0]
        # print(name)

        """read image and mask"""
        x = cv2.imread(x, cv2.IMREAD_COLOR)
        y = cv2.imread(y, cv2.IMREAD_COLOR)

        """augmentation"""
        if augment==True:
            aug = HorizontalFlip(p=1.0)
            augmented = aug(image=x, mask=y)
            x1 = augmented["image"]
            y1 = augmented["mask"]

            X = [x1]
            Y = [y1]
        else:
            X=[x]
            Y=[y]

        index=0
        for i,m in zip(X,Y):
            try:
                """ Center Cropping """
                aug = CenterCrop(H, W, p=1.0)
                augmented = aug(Image=i, Mask=m)
                i = augmented["Image"]
                m = augmented["Mask"]

            except Exception as e:
                i = cv2.resize(i, (W, H))
                m = cv2.resize(m, (W, H))


            tmp_image_name=f"{name}_{index}.png"
            tmp_mask_name=f"{name}_{index}.png"

            


            image_path = save_path + "Image/" + tmp_image_name
            mask_path = save_path + "Mask/" + tmp_mask_name

            cv2.imwrite(image_path,i)
            cv2.imwrite(mask_path,m)

            index +=1


        


"""if u want to see image and its mask so run this code"""   
    
# for x, y in zip(X,Y):
#     print(x,y)

#     x=cv2.imread(x)
#     cv2.imwrite("x.png",x)

#     y=cv2.imread(y)
#     cv2.imwrite("y.png",y)

#     break


 
if __name__=="__main__":
    np.random.seed(40)

    """Load the dataset"""
    data_path = "Segmentation"
    # x_train,x_test,y_train,y_test = load_data(data_path)
    (x_train,y_train), (x_test,y_test) = load_data(data_path)

    print("length of x_train",len(x_train))
    print("length of y_train",len(y_train))
    print("length of x_test",len(x_test))
    print("length of y_test",len(y_test))

    """create new directories for augmentation"""
    create_dir("new/train/Image/")
    create_dir("new/train/Mask/")
    create_dir("new/test/Image/")
    create_dir("new/test/Mask/")

    """data augmentation"""
    augment_data(x_train,y_train,"new/train/",augment=True) 
    augment_data(x_test,y_test,"new/test/",augment=False)  
 


    
   


