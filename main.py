import torch
from sklearn.metrics import confusion_matrix

from adni.data_loader import train_val_test_loaders
from adni.model import ADNI_MODEL

torch.cuda.empty_cache()
import sys

# This code is implemented based on following paper
# https://www.frontiersin.org/articles/10.3389/fnagi.2019.00194/full

if __name__ == "__main__":
    print(sys.argv)
    print(f'Script Name is {sys.argv[0]}')

    JOB_ID = '2' #sys.argv[1]
    ROOT_DIR = r'C:\StFX\Project\All_Files_Classified\All_Data' #r'/home/nithesh/ADNI/dataset/All_Data/'
    TRAIN_BATCH = 1  # 4
    TEST_BATCH = 1
    TRAIN_WORKERS = 0  # 4
    TEST_WORKERS = 0
    LR = 1e-4
    WD = 1e-4
    EPOCHS = 3  # 600

    """
    TRAIN_FILE = '/home/nithesh/ADNI/dataset/bagging_new/' + sys.argv[
        1] + 'train_ds_new.csv'  # '/home/nithesh/ADNI/dataset/bagging_new/1train_ds_new.csv'
    VAL_FILE = '/home/nithesh/ADNI/dataset/bagging_new/' + sys.argv[
        1] + 'val_ds_new.csv'  # '/home/nithesh/ADNI/dataset/bagging_new/1val_ds_new.csv'
    TEST_FILE = '/home/nithesh/ADNI/dataset/bagging_new/' + sys.argv[
        1] + 'test_ds_new.csv'  # '/home/nithesh/ADNI/dataset/bagging_new/1test_ds_new.csv'
    """

    TRAIN_FILE = '0train_ds_new.csv'
    VAL_FILE = '0val_ds_new.csv'
    TEST_FILE = '0test_ds_new.csv'

    train_loader, val_loader, test_loader = train_val_test_loaders(train_file=TRAIN_FILE,
                                                                   val_file=VAL_FILE,
                                                                   test_file=TEST_FILE,
                                                                   root_dir=ROOT_DIR,
                                                                   train_batch=TRAIN_BATCH,
                                                                   test_batch=TEST_BATCH,
                                                                   train_workers=TRAIN_WORKERS,
                                                                   test_workers=TEST_WORKERS
                                                                   )

    # checking for device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = ADNI_MODEL(lr=LR, wd=WD).to(device)
    print("model defined")

    model.train_and_validate(train_loader=train_loader, val_loader=val_loader, epochs=EPOCHS, job_id=JOB_ID)
    actual_label, predicted_label = model.test_on_data(test_loader)
    confusion = confusion_matrix(actual_label, predicted_label)
    print(confusion)

    torch.save(model.state_dict(), JOB_ID + 'final_best_checkpoint.model')
    print("model saved")
