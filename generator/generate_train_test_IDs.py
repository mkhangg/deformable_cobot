import os

# classes = ['tiny-can', 'small-can', 'medium-can', 'food-can,' \
#            'tall-can', 'small-cup', 'medium-cup', 'large-cup' \
#             'tiny-ball', 'small-ball', 'medium-ball', 'large-ball']
classes = ['medium-can', 'food-can', \
           'small-cup', 'medium-cup', 'large-cup', \
            'tiny-ball', 'small-ball', 'medium-ball']

train_file = open('resample_dataset/dobject8_train.txt', 'w')
test_file = open('resample_dataset/dobject8_test.txt', 'w')

for cls in classes:
    txt_files = os.listdir(f'resample_dataset/{cls}')

    n_files = len(txt_files)
    # print(f"{cls}: {n_files}")
    
    n_train = int(round(n_files*0.7, 0))
    n_test = n_files - n_train

    for i in range(n_train):
        id = txt_files[i].split('.')[0]
        train_file.write(f'{id}\n')

    for i in range(n_train, n_files):
        id = txt_files[i].split('.')[0]
        test_file.write(f'{id}\n')

train_file.close()
test_file.close()        
