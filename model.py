import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader


class ADNI_MODEL(nn.Module):
    def __init__(self, lr: float = None, wd: float = None):
        nn.Module.__init__(self)
        self.Conv_1 = nn.Conv3d(in_channels=1, out_channels=8, kernel_size=3)
        self.Conv_1_bn = nn.BatchNorm3d(8)
        self.Conv_1_mp = nn.MaxPool3d(2)
        self.res_blocks2 = nn.Sequential(*[ResBlock(8) for _ in range(2)])
        self.Conv_2 = nn.Conv3d(8, 16, 3)
        self.Conv_2_bn = nn.BatchNorm3d(16)
        self.Conv_2_mp = nn.MaxPool3d(3)
        self.res_blocks3 = nn.Sequential(*[ResBlock(16) for _ in range(2)])
        self.Conv_3 = nn.Conv3d(16, 32, 3)
        self.Conv_3_bn = nn.BatchNorm3d(32)
        self.Conv_3_mp = nn.MaxPool3d(2)
        self.res_blocks4 = nn.Sequential(*[ResBlock(32) for _ in range(2)])
        self.Conv_4 = nn.Conv3d(32, 64, 3)
        self.Conv_4_bn = nn.BatchNorm3d(64)
        self.Conv_4_mp = nn.MaxPool3d(3)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.4)
        self.dense_1 = nn.Linear(4800, 128)
        self.dropout2 = nn.Dropout(p=0.4)
        self.dense_2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()
        self.optimizer = Adam(self.parameters(), lr=lr, weight_decay=wd)

    def forward(self, x):
        x = self.relu(self.Conv_1_bn(self.Conv_1(x)))
        x = self.Conv_1_mp(x)
        x = self.res_blocks2(x)
        x = self.relu(self.Conv_2_bn(self.Conv_2(x)))
        x = self.Conv_2_mp(x)
        x = self.res_blocks3(x)
        x = self.relu(self.Conv_3_bn(self.Conv_3(x)))
        x = self.Conv_3_mp(x)
        x = self.res_blocks4(x)
        x = self.relu(self.Conv_4_bn(self.Conv_4(x)))
        x = self.Conv_4_mp(x)
        x = x.view(x.size(0), -1)
        x = self.dropout1(x)
        x = self.relu(self.dense_1(x))
        x = self.dropout2(x)
        x = self.dense_2(x)
        x = self.sigmoid(x)
        return x

    def train_and_validate(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int, job_id : int):

        for epoch in range(epochs):
            # training on training dataset
            train_accuracy, train_loss = self.train_on_data(train_loader)

            # Evaluation on testing dataset
            val_accuracy, val_loss = self.validate_on_data(val_loader)

            # To save the model after each epoch, if validation accuracy is > 96%
            if (val_accuracy >= 0.96):
                torch.save(self.state_dict(), job_id + 'best_checkpoint_' + str(epoch) + '.model')

            print('Epoch: ' + str(epoch) + ' Train Loss: ' + str(train_loss) + ' Val Loss: ' + str(
                val_loss) + ' Train Accuracy: ' + str(
                train_accuracy) + ' Val Accuracy: ' + str(val_accuracy))

    def train_on_data(self, train_loader: DataLoader):
        self.train()
        train_accuracy = 0.0
        train_loss = 0.0

        for i, (images, labels) in enumerate(train_loader):
            images = images.type(torch.FloatTensor)
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()

            self.optimizer.zero_grad()
            outputs = self(images)
            outputs = outputs.reshape(-1)
            outputs = outputs.type(torch.FloatTensor)
            outputs = outputs.cuda()
            loss = F.binary_cross_entropy(outputs, labels)

            loss.backward()
            self.optimizer.step()

            train_loss += loss.cpu().data * images.size(0)

            preds = torch.round(outputs)
            train_accuracy += int(torch.sum(preds.data == labels.data))

        train_accuracy = train_accuracy / len(train_loader)
        train_loss = train_loss / len(train_loader)
        return train_accuracy, train_loss

    def validate_on_data(self, val_loader: DataLoader):
        self.eval()
        val_loss = 0.0
        val_accuracy = 0.0
        with torch.no_grad():
            for i, (images, labels) in enumerate(val_loader):
                images = images.type(torch.FloatTensor)
                if torch.cuda.is_available():
                    images = images.cuda()
                    labels = labels.cuda()

                outputs = self(images)
                outputs = outputs.reshape(-1)
                outputs = outputs.type(torch.FloatTensor)
                outputs = outputs.cuda()
                labels = labels.type(torch.FloatTensor)
                labels = labels.cuda()
                loss = F.binary_cross_entropy(outputs, labels)
                val_loss += loss.cpu().data * images.size(0)

                preds = torch.round(outputs)
                val_accuracy += int(torch.sum(preds.data == labels.data))

        val_accuracy = val_accuracy / len(val_loader)
        val_loss = val_loss / len(val_loader)
        return val_accuracy, val_loss

    def test_on_data(self, test_loader: DataLoader):
        # model testing
        self.eval()
        actual_label = []
        predicted_label = []

        test_accuracy = 0.0
        with torch.no_grad():
            for i, (images, labels) in enumerate(test_loader):
                images = images.type(torch.FloatTensor)
                labels = labels.type(torch.FloatTensor)
                if torch.cuda.is_available():
                    images = images.cuda()
                    labels = labels.cuda()

                outputs = self(images)
                outputs = outputs.reshape(-1)
                outputs = outputs.type(torch.FloatTensor)
                outputs = outputs.cuda()

                outputs = torch.round(outputs)
                index1 = labels.cpu().data.numpy()
                actual_label.append(index1)
                index2 = outputs.cpu().data.numpy()
                predicted_label.append(index2)
                test_accuracy += int(torch.sum(outputs == labels.data))

        test_accuracy = test_accuracy / len(test_loader)

        print(' Test Accuracy: ' + str(test_accuracy))
        print(actual_label)
        print(predicted_label)
        return actual_label, predicted_label

class ResBlock(nn.Module):
    def __init__(self, channel: int, ratio=4):
        super(ResBlock, self).__init__()

        self.conv_lower = nn.Sequential(
            nn.Conv3d(channel, channel, 3, padding=1, bias=False),
            nn.BatchNorm3d(channel),
            nn.ReLU(inplace=True)
        )

        self.conv_upper = nn.Sequential(
            nn.Conv3d(channel, channel, 3, padding=1, bias=False),
            nn.BatchNorm3d(channel)
        )

        self.ca = ChannelAttention(channel, ratio)
        self.sa = SpatialAttention()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        path = self.conv_lower(x)
        path = self.conv_upper(path)
        path = self.ca(path) * path
        path = self.sa(path) * path

        return self.relu(path + x)


class ChannelAttention(nn.Module):
    def __init__(self, channel: int, ratio: int):
        super(ChannelAttention, self).__init__()

        self.shared_mlp = nn.Sequential(
            nn.Conv3d(channel, channel // ratio, 1, padding=0, bias=False),  # channel // ratio
            nn.ReLU(inplace=True),
            nn.Conv3d(channel // ratio, channel, 1, padding=0, bias=False)  # channel // ratio
        )

        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat_avg = self.shared_mlp(self.avg_pool(x))
        feat_max = self.shared_mlp(self.max_pool(x))

        return self.sigmoid(feat_avg + feat_max)


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()

        self.conv = nn.Conv3d(2, 1, 7, padding=3, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat_avg = torch.mean(x, dim=1, keepdim=True)
        feat_max = torch.max(x, dim=1, keepdim=True)[0]

        feature = torch.cat((feat_avg, feat_max), dim=1)

        return self.sigmoid(self.conv(feature))
