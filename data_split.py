import os
import random
import shutil

if __name__ == '__main__':
    # Setup dir paths
    DATA_PATH = "data"
    TRAIN_PATH = os.path.join(DATA_PATH, "train")
    TEST_PATH = os.path.join(DATA_PATH, "test")

    classes = ("cat", "dog", "snake")
    test_split = 0.2

    # Make test directories
    for cls in classes:
        os.makedirs(os.path.join(TEST_PATH, cls), exist_ok=True)

    for cls in classes:
        CLASS_TRAIN_PATH = os.path.join(TRAIN_PATH, cls)
        CLASS_TEST_PATH = os.path.join(TEST_PATH, cls)

        images = os.listdir(CLASS_TRAIN_PATH)
        random.shuffle(images)

        n_test = int(test_split*len(images))
        test_images = images[:n_test]
        
        for image in test_images:
            src = os.path.join(CLASS_TRAIN_PATH, image)
            dst = os.path.join(CLASS_TEST_PATH, image)
            shutil.move(src, dst)

# Checking train/test split
for cls in classes:
    CLASS_TRAIN_PATH = os.path.join(TRAIN_PATH, cls)
    CLASS_TEST_PATH = os.path.join(TEST_PATH, cls)

    train_images = os.listdir(CLASS_TEST_PATH)
    test_images = os.listdir(CLASS_TRAIN_PATH)

    print(f"There are {len(train_images)} train images and {len(test_images)} test images of {cls}")