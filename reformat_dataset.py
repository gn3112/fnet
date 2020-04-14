import os
from shutil import copyfile, copytree
from random import randint

path = 'fruits-360_dataset/fruits-360/Test/'
if not os.path.exists('new_fruit_dataset/Test'):
    os.makedirs('new_fruit_dataset/Test')

classes = [f for f in os.listdir(path) if not f.startswith('.')]
print(classes)
new_classes = []
for class_ in classes:
    current_class_short = class_.split()[0]
    for new_class in new_classes:
        if new_class == current_class_short:
            new_class_dir = 'new_fruit_dataset/Test/{}'.format(current_class_short)
            class_files = os.listdir('fruits-360_dataset/fruits-360/Test/{}'.format(class_))
            for class_file in class_files:
                new_class_file = str(randint(0,1000)) + '_' + class_file
                copyfile('fruits-360_dataset/fruits-360/Test/{}/{}'.format(class_,class_file),
                            'new_fruit_dataset/Test/{}/{}'.format(current_class_short,new_class_file))
            break

    if len(new_classes)!=0 and new_class == current_class_short:
        continue
    else:
        # os.makedirs('new_fruit_dataset/Test/{}'.format(current_class_short))
        new_classes.append(current_class_short)
        copytree('fruits-360_dataset/fruits-360/Test/{}'.format(class_),
            'new_fruit_dataset/Test/{}'.format(current_class_short))
