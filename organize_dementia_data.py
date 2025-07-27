import os
import csv
import shutil

# Paths
GROUNDTRUTH_CSV = 'dementia samples/training-groundtruth.csv'
AUDIO_SRC_DIR = 'dementia samples/train'
DATA_DIR = 'data'
CONTROL_DIR = os.path.join(DATA_DIR, 'control')
DEMENTIA_DIR = os.path.join(DATA_DIR, 'dementia')

# Ensure target directories exist
os.makedirs(CONTROL_DIR, exist_ok=True)
os.makedirs(DEMENTIA_DIR, exist_ok=True)

def main():
    with open(GROUNDTRUTH_CSV, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            audio_id = row['adressfname'].strip().replace('"', '')
            label = row['dx'].strip().replace('"', '')
            if not audio_id:
                continue
            src_file = os.path.join(AUDIO_SRC_DIR, f'{audio_id}.mp3')
            if not os.path.exists(src_file):
                print(f'File not found: {src_file}')
                continue
            if label == 'Control':
                dst_dir = CONTROL_DIR
            elif label == 'ProbableAD':
                dst_dir = DEMENTIA_DIR
            else:
                print(f'Unknown label for {audio_id}: {label}')
                continue
            dst_file = os.path.join(dst_dir, f'{audio_id}.mp3')
            shutil.copy2(src_file, dst_file)
            print(f'Copied {src_file} -> {dst_file}')
    print('Data organization complete.')

if __name__ == '__main__':
    main() 