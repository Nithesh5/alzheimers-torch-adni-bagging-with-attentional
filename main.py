import torch
from sklearn.metrics import confusion_matrix

from adni.data_loader import train_val_test_loaders
from adni.model import ADNI_MODEL

torch.cuda.empty_cache()
import sys

"""
only base model is implemented based on following paper
https://arxiv.org/abs/1807.06521
Attentional layers are implemented based on following paper
https://www.frontiersin.org/articles/10.3389/fnagi.2019.00194/full
"""

if __name__ == "__main__":
    print(sys.argv)

    JOB_ID = sys.argv[1]
    ROOT_DIR = r'/home/nithesh/ADNI/dataset/All_Data/'
    TRAIN_BATCH = 4
    TEST_BATCH = 1
    TRAIN_WORKERS = 4
    TEST_WORKERS = 0
    LR = 1e-4
    WD = 1e-4
    EPOCHS = 600

    TRAIN_FILE = '/home/nithesh/ADNI/dataset/bagging_new/' + JOB_ID + 'train_ds_new.csv'
    VAL_FILE = '/home/nithesh/ADNI/dataset/bagging_new/' + JOB_ID + 'val_ds_new.csv'
    TEST_FILE = '/home/nithesh/ADNI/dataset/bagging_new/' + JOB_ID + 'test_ds_new.csv'

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
