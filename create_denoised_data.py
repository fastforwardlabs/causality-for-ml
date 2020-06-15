'''
Create a subset dataset based on the original iWildCam data

'''
import pandas as pd
import os
import shutil

INPUT_DIR = "/datapool/wildcam/iWildCam/train"
OUTPUT_DIR = "./data/wildcam_denoised"

if __name__ == '__main__':
    filenames_df = pd.read_json("./data/train_test_filenames.json")
    print(filenames_df.head())

    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)
        test_data_dir = os.path.join(OUTPUT_DIR, 'test')
        os.mkdir(test_data_dir)
        for animal in set(filenames_df["category"].tolist()):
            os.mkdir(os.path.join(test_data_dir, animal))
            subset_df = filenames_df.loc[(filenames_df['category'] == animal)
                    & (filenames_df['location'] == 130)]
            for filename in subset_df['filename'].tolist():
                shutil.copy2(os.path.join(INPUT_DIR, filename),
                        os.path.join(test_data_dir, animal))
        for train_env in (43, 46):
            train_loc = os.path.join(OUTPUT_DIR, "train_" + str(train_env))
            os.mkdir(train_loc)
            for animal in set(filenames_df["category"].tolist()):
                os.mkdir(os.path.join(train_loc, animal))
                subset_df = filenames_df.loc[(filenames_df['category'] ==
                    animal) & (filenames_df['location'] == train_env)]
                for filename in subset_df['filename'].tolist():
                    shutil.copy2(os.path.join(INPUT_DIR, filename),
                            os.path.join(train_loc, animal))
    else:
        print("Warning: output dir {} already exists".format(OUTPUT_DIR))
