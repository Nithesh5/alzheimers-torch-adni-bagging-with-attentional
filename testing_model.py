import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from torch.utils.data import DataLoader

from adni.data_loader import ADNIDataloaderAllData
from adni.model import ADNI_MODEL

TEST_FILE = '0test_ds_new.csv'
ROOT_DIR = r'C:\StFX\Project\All_Files_Classified\All_Data'
LR = 1e-4
WD = 1e-4

# Test dataset
test_ds = pd.read_csv(TEST_FILE)

print("len of train and val and test")
print(len(test_ds))

test_dataset = ADNIDataloaderAllData(df=test_ds,
                                     root_dir=ROOT_DIR,
                                     transform=None)

test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=0)

# checking for device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# loading all 10 best models
model1 = ADNI_MODEL(lr=LR, wd=WD).to(device)
model1.load_state_dict(torch.load('1best_checkpoint_232.model'))  #

model2 = ADNI_MODEL(lr=LR, wd=WD).to(device)
model2.load_state_dict(torch.load('2best_checkpoint_113.model'))  #

model3 = ADNI_MODEL(lr=LR, wd=WD).to(device)
model3.load_state_dict(torch.load('3best_checkpoint_129.model'))  #

model4 = ADNI_MODEL(lr=LR, wd=WD).to(device)
model4.load_state_dict(torch.load('4best_checkpoint_162.model'))  #

model5 = ADNI_MODEL(lr=LR, wd=WD).to(device)
model5.load_state_dict(torch.load('5best_checkpoint_108.model'))  #

model6 = ADNI_MODEL(lr=LR, wd=WD).to(device)
model6.load_state_dict(torch.load('6best_checkpoint_167.model'))  #

model7 = ADNI_MODEL(lr=LR, wd=WD).to(device)
model7.load_state_dict(torch.load('7best_checkpoint_75.model'))  #

model8 = ADNI_MODEL(lr=LR, wd=WD).to(device)
model8.load_state_dict(torch.load('8best_checkpoint_217.model'))  #

model9 = ADNI_MODEL(lr=LR, wd=WD).to(device)
model9.load_state_dict(torch.load('9best_checkpoint_169.model'))  #

models = [model1, model2, model3, model4, model5, model6, model7, model8, model9]

# model testing
for x in models:
    x.eval()

actual_label = []
predicted_label = []

test_accuracy = 0.0

with torch.no_grad():
    for i, (images, labels) in enumerate(test_loader):
        images = images.type(torch.FloatTensor)
        labels = labels.type(torch.FloatTensor)
        outputs = []
        summed = []

        output_mode = 0
        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()

        for x in models:
            outputs.append(x(images))

        for x in outputs:
            reshape = x.reshape(-1)
            summed.append(torch.round(reshape))

        sum = np.sum(summed) / len(models)
        output_mode = torch.round(sum)

        index1 = labels.cpu().data.numpy()
        actual_label.append(index1)

        predicted_label.append(output_mode.cpu().data.numpy())
        test_accuracy += int(torch.sum(output_mode == labels.data))

test_accuracy = test_accuracy / len(test_loader)

print(' Test Accuracy: ' + str(test_accuracy))

# refered from official site
# https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_display_object_visualization.html#sphx-glr-auto-examples-miscellaneous-plot-display-object-visualization-py

cm = confusion_matrix(actual_label, predicted_label)
cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['AD', 'CN']).plot()
plt.show()

fpr, tpr, _ = roc_curve(actual_label, predicted_label)
roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
plt.show()

print("done")