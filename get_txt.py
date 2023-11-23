import os

# your img path
path = r'img_path'

crab_list = sorted(os.listdir(path))[:-20]

output_train_file_path = 'img_path/train.txt'
with open(output_train_file_path, 'w') as file:
    for crab in crab_list:
        crab_samples = sorted(os.listdir(path+'/'+crab))
        for crab_sample in crab_samples:
            file.write(crab+'/'+crab_sample[-8:-4] + '\n')

print(f"File paths saved to {output_train_file_path}")


crab_list = sorted(os.listdir(path))[-20:]
output_val_file_path = 'img_path/val.txt'
with open(output_val_file_path, 'w') as file:
    for crab in crab_list:
        crab_samples = sorted(os.listdir(path+'/'+crab))
        for crab_sample in crab_samples:
            file.write(crab+'/'+crab_sample[-8:-4] + '\n')

print(f"File paths saved to {output_val_file_path}")

