# -*- encoding:utf-8 -*-
import os
import sys
import os.path
#
# def ChangeFileName(folder, file_list):
#     for file_line in file_list:
#         old_file_name = file_line
#         new_file_name = file_line.replace(".jpg.txt", ".txt")
#         print "new: " + new_file_name
#         print "old: " + old_file_name
#         if new_file_name != old_file_name:
#             print "file_name:" + old_file_name
#             print "new_file_name:" + new_file_name
#             os.rename(os.path.join(folder, file_line), os.path.join(folder, new_file_name))


if __name__ == "__main__":
    piclist=[]
    for filename in os.listdir('/home/m1/Downloads/cpytest_origin/'):
        if os.path.splitext(filename)[1] == '.jpg':
            piclist.append(filename)
            print(filename)
    i=1
    for pic in piclist:
        os.rename(os.path.join('/home/m1/Downloads/cpytest_origin/', pic), os.path.join('/home/m1/Downloads/cpytest_origin/', str(i)+"x.jpg"))
        i=i+1


#   0_all.png +3 ->  3_all.png
# if __name__ == "__main__":
#     piclist=[]
#     for filename in os.listdir('/home/m1/Downloads/pic'):
#         if os.path.splitext(filename)[1] == '.png':
#             if filename[-8:] == '_all.png':
#                 piclist.append(filename)
#                 print(filename)
#
#     for pic in piclist:
#         a = int(pic[:-8])+3
#         pic2 = str(a) + pic[-8:]
#         os.rename(os.path.join('/home/m1/Downloads/pic/', pic), os.path.join('/home/m1/Downloads/pic/', pic2))

