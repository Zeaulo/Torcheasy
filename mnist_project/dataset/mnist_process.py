#!usr/bin/env python3
# *-* coding: utf-8 *-*

import numpy as np

def process(dataset, save_name):
    data_name = f'{dataset}_data'
    f = open(data_name, "rb")
    f.read(16) # 16 bytes of useless information in front
    images = []

    label_name = f'{dataset}_label'
    l = open(label_name, "rb")
    l.read(8) # 8 bytes of useless information in front
    labels = []

    lenth = {'train':60000, 'test':10000}
    
    for a in range(lenth[dataset]):
        image = np.zeros([28, 28])
        label = ord(l.read(1))
        labels.append(label)
        for i in range(28):
            for j in range(28):
                image[i][j] = ord(f.read(1))
        
        images.append(image)

    images = np.array(images)
    labels = np.array(labels)

    np.save(f'{save_name}_data' , images)
    np.save(f'{save_name}_label', labels)
    print(f'Finished {dataset}.')
    
if __name__ == '__main__':
    process('train', 'train')
    process('test' , 'val')