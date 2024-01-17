###IMPORTANT NOTE: the split and fill_in code, may have directories slightly wrong, as I've moved things around a bit and coded these sections
### by themselves. Not yet sutible to run without some changes (currently do it section by section, changing directories as needed)

from ultralytics.data.converter import convert_coco
import os
import zipfile
import shutil
import glob
import random

#export from CVAT as COCO1.1
#comes in as ziped file with annotations/instances_default.json
file_name = 'aloripes_dec_2023_500_ann.zip'
save_dir = '/home/java/Java/data/aloripes_dec_2023_500_ann'
download_dir = '/home/java/Downloads'
split = True #got to get images in dir before split
fill_in = False #if want to fill in blank text files

# convert coco to yolo
with zipfile.ZipFile(os.path.join(download_dir, file_name), 'r') as zip_ref:
   zip_ref.extractall(download_dir)
    #annotation folder in downloads

downloaded_labels_dir = os.path.join(download_dir, 'annotations')
lable_dir = os.path.join(save_dir, 'labels')
coco_labels = os.path.join(lable_dir, 'default')

convert_coco(labels_dir=downloaded_labels_dir, save_dir=save_dir,
                 use_segments=True, use_keypoints=False, cls91to80=False)

for filename in os.listdir(coco_labels):
    source = os.path.join(coco_labels, filename)
    destination = os.path.join(lable_dir, filename)
    if os.path.isfile(source):
        shutil.move(source, destination)

#if images with no annotations
#need to add in a bank text file for labels
if fill_in:
    images_folder = save_dir+'images'
    labels_folder = save_dir+'labels'
    image_files = [os.path.splitext(file)[0] for file in os.listdir(images_folder)]
    label_files = [os.path.splitext(file)[0] for file in os.listdir(labels_folder)]
    missing_files = [file for file in image_files if file not in label_files]
    for file_name in missing_files:
        with open(os.path.join(labels_folder, file_name+'.txt'), 'w') as file:
            pass
    print ("blank text files added")

#if want to split train, val, test data
#assume images in same folder as labels
save_dir = '/home/java/Java/data/cslics_aloripedes_n_amtenuis_jan'
if split:
    train_ratio = 0.8
    test_ratio = 0.1
    valid_ratio = 0.1
    def check_ratio(test_ratio,train_ratio,valid_ratio):
        if(test_ratio>1 or test_ratio<0): ValueError(test_ratio,f'test_ratio must be > 1 and test_ratio < 0, test_ratio={test_ratio}')
        if(train_ratio>1 or train_ratio<0): ValueError(train_ratio,f'train_ratio must be > 1 and train_ratio < 0, train_ratio={train_ratio}')
        if(valid_ratio>1 or valid_ratio<0): ValueError(valid_ratio,f'valid_ratio must be > 1 and valid_ratio < 0, valid_ratio={valid_ratio}')
        if not((train_ratio+test_ratio+valid_ratio)==1): ValueError("sum of train/val/test ratio must equal 1")
    check_ratio(test_ratio,train_ratio,valid_ratio)

    imagelist = glob.glob(os.path.join(save_dir,'images', '*.jpg'))
    txtlist = glob.glob(os.path.join(save_dir, 'labels', '*.txt'))
    txtlist.sort()
    imagelist.sort()
    imgno = len(txtlist) 
    noleft = imgno

    validimg, validtext, testimg, testtext = [], [], [], []

    # function to seperate files into different lists randomly while retaining the same .txt and .jpg name in the specific type of list
    def seperate_files(number,imglist,textlist,noleft):
        for i in range(int(number)):
            r = random.randint(0, noleft)
            imglist.append(imagelist[r])
            textlist.append(txtlist[r])
            txtlist.remove(txtlist[r])
            imagelist.remove(imagelist[r])
            noleft -= 1
        return noleft

    #pick some random files
    noleft = seperate_files(imgno*valid_ratio,validimg,validtext,noleft)
    seperate_files(imgno*test_ratio,testimg,testtext,noleft)

    # function to preserve symlinks of src file, otherwise default to copy
    def copy_link(src, dst):
        if os.path.islink(src):
            linkto = os.readlink(src)
            os.symlink(linkto, os.path.join(dst, os.path.basename(src)))
        else:
            shutil.copy(src, dst)
    # function to make sure the directory is empty
    def clean_dirctory(savepath):
        if os.path.isdir(savepath):
            shutil.rmtree(savepath)
        os.makedirs(savepath, exist_ok=True)
    # function to move a list of files, by cleaning the path and copying and preserving symlinks
    def move_file(filelist,savepath,second_path):
        output_path = os.path.join(savepath, second_path)
        #clean_dirctory(output_path)
        os.makedirs(output_path, exist_ok=True)
        for i, item in enumerate(filelist):
            copy_link(item, output_path)

    move_file(txtlist,save_dir,'labels/train')
    move_file(imagelist,save_dir,'images/train')
    move_file(validtext,save_dir,'labels/valid')
    move_file(validimg,save_dir,'images/valid')
    move_file(testtext,save_dir,'labels/test')
    move_file(testimg,save_dir,'images/test')

    print("split complete")