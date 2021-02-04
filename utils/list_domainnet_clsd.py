import os
import sys

source = sys.argv[1]
target = sys.argv[2]
p_path = os.path.join('/research/masaito/multisource_data/few_shot_DA_data', source)

p_path2 = os.path.join('/research/masaito/multisource_data/few_shot_DA_data', 'real')

dir_list = os.listdir(p_path2)
dir_list.sort()

source_list = dir_list # + unshared_list[:10]
target_list = dir_list#
print(source_list)
print(target_list)
path_source = "./source_%s_cls.txt"%('d'+source+'125')
path_target = "./target_%s_cls.txt"%('d'+target+'125')


write_source = open(path_source,"w")
write_target = open(path_target,"w")
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
for k, direc in enumerate(target_list):
    if not '.txt' in direc:
        files = os.listdir(os.path.join(p_path, direc))
        for i, file in enumerate(files):
            file_name = os.path.join(p_path, direc, file)
            if direc in source_list:
                class_name = direc
                write_target.write('%s %s\n' % (file_name, source_list.index(class_name)))
            elif direc in target_list:
                file_name = os.path.join(p_path, direc, file)
                write_target.write('%s %s\n' % (file_name, len(source_list)))


