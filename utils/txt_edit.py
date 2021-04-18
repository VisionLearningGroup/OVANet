import sys
import os
files = os.listdir(sys.argv[1])

for file in files:
    file_new = open(os.path.join(sys.argv[2], file), 'w')
    file_old = open(os.path.join(sys.argv[1], file), 'r')
    lines = [line.strip() for line in file_old.readlines()]
    for line in lines:
        name = line.split(' ')[0]
        ind = line.split(' ')[1]
        name = name.replace('data/cross_domain_few/', 'data/')
        file_new.write('%s %s\n'%(name, ind))


