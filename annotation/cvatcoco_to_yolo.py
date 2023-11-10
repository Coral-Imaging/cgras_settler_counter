from ultralytics.data.converter import convert_coco
import os
import zipfile
import shutil
import glob
import random

#export from CVAT as COCO1.1
#comes in as ziped file with annotations/instances_default.json
file_name = 'coco_5_images_polygons.zip'
save_dir = '/home/java/Java/data/new_data'
download_dir = '/home/java/Downloads'
split = False #got to get images in dir before split

with zipfile.ZipFile(os.path.join('/home/java/Downloads', file_name), 'r') as zip_ref:
   zip_ref.extractall('/home/java/Downloads')
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

#if want to split train, val, test data
#assume images in same folder as labels
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


    imagelist = glob.glob(os.path.join(save_dir, '*.PNG'))
    txtlist = glob.glob(os.path.join(save_dir, '*.txt'))
    txtlist.sort()
    imagelist.sort()
    imgno = len(txtlist) 

    validimg = []
    validtext = []
    testimg = []
    testtext = []
    def seperate_files(number,imglist,textlist):
        for i in range(int(number)):
            r = random.randint(0, len(textlist))
            imglist.append(imagelist[r])
            textlist.append(txtlist[r])
            txtlist.remove(txtlist[r])
            imagelist.remove(imagelist[r])

    #pick some random files
    seperate_files(imgno*valid_ratio,validimg,validtext)
    seperate_files(imgno*test_ratio,testimg,testtext)

    # function to preserve symlinks of src file, otherwise default to copy
    def copy_link(src, dst):
        if os.path.islink(src):
            linkto = os.readlink(src)
            os.symlink(linkto, os.path.join(dst, os.path.basename(src)))
        else:
            shutil.copy(src, dst)

    def clean_dirctory(savepath):
        if os.path.isdir(savepath):
            shutil.rmtree(savepath)
        os.makedirs(savepath, exist_ok=True)

    def move_file(filelist,savepath,second_path):
        output_path = os.path.join(savepath, second_path)
        clean_dirctory(output_path)
        os.makedirs(output_path, exist_ok=True)
        for i, item in enumerate(filelist):
            copy_link(item, output_path)

    move_file(txtlist,save_dir,'train/labels')
    move_file(imagelist,save_dir,'train/images')
    move_file(validtext,save_dir,'valid/labels')
    move_file(validimg,save_dir,'valid/images')
    move_file(testimg,save_dir,'test/images')
    move_file(testtext,save_dir,'test/labels')