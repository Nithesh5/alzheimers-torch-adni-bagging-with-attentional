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

models_list = ['1best_checkpoint_232.model', '2best_checkpoint_113.model', '3best_checkpoint_129.model',
               '4best_checkpoint_162.model', '5best_checkpoint_108.model', '6best_checkpoint_167.model',
               '7best_checkpoint_75.model', '8best_checkpoint_217.model', '9best_checkpoint_169.model']

models = {}
for idx, val in enumerate(models_list):
    modelname = "model_" + str(idx)
    model = ADNI_MODEL(lr=LR, wd=WD).to(device)
    model.load_state_dict(torch.load(val))
    models[modelname] = model

for x in models.values():
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

        for x in models.values():
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
