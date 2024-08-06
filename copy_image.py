import os
import shutil


def copy_n_file(dir,destination,num):

    filelist = os.listdir(dir)
      
    # Destination path
    if not os.path.exists(destination):
            os.makedirs(destination)

    n=0
    i=0
    j=1
    for j in filelist:
        source=filelist[n]
        for i in range(num):
            new_file = os.path.join(destination, '%03d_'%(i) + os.path.basename(source))
            shutil.copy(dir+source, new_file)
        n+=1


copy_n_file('Image/positive/Training/','Dataset/training_set/positive/',150)
copy_n_file('Image/negative/Training/','Dataset/training_set/negative/',500)
copy_n_file('Image/positive/Testing/','Dataset/test_set/positive/',150)
copy_n_file('Image/negative/Testing/','Dataset/test_set/negative/',500)