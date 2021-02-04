import os
import sys

source = sys.argv[1]
target = sys.argv[2]
p_path = os.path.join('/research/masaito/multisource_data/few_shot_DA_data', source)

p_path2 = os.path.join('/research/masaito/multisource_data/few_shot_DA_data', 'real')

dir_list = os.listdir(p_path2)
dir_list.sort()

source_list = dir_list[:80] # + unshared_list[:10]
target_list = dir_list[:100]#
print(source_list)
print(target_list)
path_source = "../txt/source_%s_open.txt"%('d'+source+'125')
per_class = 5
path_target_few = "../txt/target_%s_open_few_%d.txt"%('d'+target+'125', per_class)
path_target_unl = "../txt/target_%s_open_unl.txt"%('d'+target+'125')


write_source = open(path_source,"w")
write_target_few = open(path_target_few,"w")
write_target_unl = open(path_target_unl,"w")
for k, direc in enumerate(source_list):
    if not '.txt' in direc:
        files = os.listdir(os.path.join(p_path, direc))
        for i, file in enumerate(files):
            if direc in source_list:
                class_name = direc
                file_name = os.path.join(p_path, direc, file)
                write_source.write('%s %s\n' % (file_name, source_list.index(class_name)))
            else:
                continue
p_path = os.path.join('/research/masaito/multisource_data/few_shot_DA_data', target)
dir_list = os.listdir(p_path)
dir_list.sort()
print(target_list)
print(len(target_list))
import random
for k, direc in enumerate(dir_list):
    if not '.txt' in direc:
        files = os.listdir(os.path.join(p_path, direc))
        random.shuffle(files)
        for i, file in enumerate(files):
            file_name = os.path.join(p_path, direc, file)
            if direc in target_list and i < per_class:
                class_name = direc
                write_target_few.write('%s %s\n' % (file_name, target_list.index(class_name)))
            elif direc in target_list:
                write_target_unl.write('%s %s\n' % (file_name, target_list.index(class_name)))
            else:
                write_target_unl.write('%s %s\n' % (file_name, len(target_list)))


