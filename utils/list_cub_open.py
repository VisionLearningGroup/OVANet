import os
import random
import sys
p_path = "/research/masaito/cub/CUB_200_2011/images"#os.path.join('/research/masaito/office/', source,'images')
dir_list = os.listdir(p_path)
print(dir_list)
path_source = "./cub_source.txt"
#path_target = "./cub_labeled_sp1.txt"
path_target_unl = "./cub_unl.txt"

#write_source = open(path_source,"w")
write_target = open(path_source,"w")
write_target_unl = open(path_target_unl,"w")
print(dir_list)
for k, direc in enumerate(dir_list):
    if not '.txt' in direc:
        files = os.listdir(os.path.join(p_path, direc))
        for i, file in enumerate(files):
            file_name = os.path.join(p_path, direc, file)
            file_name = os.path.join(p_path, direc, file)
            if i < int(len(files)/2) and k < 150:
                write_target.write('%s %s\n' % (file_name, k))
            else:
                if k < 150:
                    write_target_unl.write('%s %s\n' % (file_name, k))
                else:
                    write_target_unl.write('%s %s\n' % (file_name, 150))


