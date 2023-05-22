"""
Purpose:Split the HAM10k,COVID-19,PBC datasets into the Non-IID settings.

"""
import torch
import os
import random
import shutil
import csv
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import pandas as pd
import time


##################################################################################################
# Dataset： ham10k
# Save the ham10 dataset
def save_ham10k_dataset():
    st_time = time.time()

    csv_path = './dataset/ham10k/HAM10000_metadata.csv'
    img_file = pd.read_csv(csv_path)
    input_path = './dataset/ham10k/HAM_10000'
    output_path = './dataset/ham10k/my_ham_dataset'
    if os.path.exists(output_path):
        os.system('rm -rf {}'.format(output_path))
    os.mkdir(output_path)

    img_names = [f for f in os.listdir(input_path)]
    #
    img = img_file['image_id'].tolist()
    label = img_file['dx'].tolist()
    for i in range(len(img)):
        image = os.path.join(input_path, img[i] + '.jpg')
        out_dir = os.path.join(output_path, label[i])
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        shutil.copy(image, out_dir)
        # print(image)
    print('Finish time:{:.4f}'.format(time.time() - st_time))

##################################################################################################
# Dataset： ham10k_2clients
# 1. Split samples per class referring to nun_test = 23
# 2. Split training set into 2 clients based on the sites in csv file
def split_ham10k_2clients():
    ham_root = './dataset/ham10k'
    ham_data_root = os.path.join(ham_root, 'my_ham_dataset')
    labels = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
    for lbl in labels:
        lbl_p = os.path.join(ham_data_root, lbl)
        imgs = [f for f in os.listdir(lbl_p) if '.jpg' in f]
        print('class: {}, num: {}'.format(lbl, len(imgs)))

    # # Create blank train test folder in ham1k_2clients folder
    dst_root = os.path.join(ham_root, 'ham10k_2clients')
    if os.path.exists(dst_root):
        os.system('rm -rf {}'.format(dst_root))
    os.mkdir(dst_root)

    for fd in ['train', 'test']:
        fd_p = os.path.join(dst_root, fd)
        os.mkdir(fd_p)

    # Copy training images and testing images
    for lbl in labels:
        lbl_p = os.path.join(ham_data_root, lbl)
        imgs = [f for f in os.listdir(lbl_p) if '.jpg' in f]

        for _ in range(10):
            random.shuffle(imgs)

        num_test = 23
        num_train = len(imgs) - num_test

        train_imgs = imgs[: num_train]
        test_imgs = imgs[num_train:]

        # Copy images to train folder
        train_root = os.path.join(dst_root, 'train', lbl)
        if not os.path.exists(train_root):
            os.mkdir(train_root)
        for img in train_imgs:
            img_p = os.path.join(lbl_p, img)
            shutil.copy2(img_p, train_root)

        # Copy images to test folder
        test_root = os.path.join(dst_root, 'test', lbl)
        if not os.path.exists(test_root):
            os.mkdir(test_root)
        for img in test_imgs:
            img_p = os.path.join(lbl_p, img)
            shutil.copy2(img_p, test_root)

    # Check images num in train / test folder
    for lbl in labels:
        lbl_p = os.path.join(ham_data_root, lbl)
        imgs = [f for f in os.listdir(lbl_p) if '.jpg' in f]

        train_lbl_p = os.path.join(dst_root, 'train', lbl)
        train_imgs = [f for f in os.listdir(train_lbl_p) if '.jpg' in f]

        test_lbl_p = os.path.join(dst_root, 'test', lbl)
        test_imgs = [f for f in os.listdir(test_lbl_p) if '.jpg' in f]

        print(len(imgs), len(train_imgs) + len(test_imgs))

    # Split training image and testing images into 2 clients based on the csv file
    sites = ['vidir', 'rosendahl']
    # sourcs = ['vidir_modern', 'vidir_molemax', 'rosendahl', 'vienna_dias']
    ham_csv_path = os.path.join(ham_root, 'HAM10000_metadata.csv')
    data = []
    with open(ham_csv_path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        next(reader)
        for row in reader:
            data.append(row)
    print(len(data))

    # Create blank clients folder
    client_root = os.path.join(dst_root, 'clients')
    if os.path.exists(client_root):
        os.system('rm -rf {}'.format(client_root))
    os.mkdir(client_root)
    for client_id in range(len(sites)):
        client_p = os.path.join(client_root, str(client_id))
        if not os.path.exists(client_p):
            os.mkdir(client_p)

        for lbl in labels:
            lbl_p = os.path.join(client_p, lbl)
            if not os.path.exists(lbl_p):
                os.mkdir(lbl_p)

    # image_id: image_path dict in training set
    img_id_path_dict = {}
    for lbl in labels:
        p = os.path.join(dst_root, 'train', lbl)
        imgs = [f for f in os.listdir(p) if '.jpg' in f]
        for img in imgs:
            img_id = img.replace('.jpg', '').strip()
            img_p = os.path.join(p, img)
            img_id_path_dict[img_id] = img_p

    train_count = 0
    for item in data:
        img_id = item[1]
        src = item[-1]
        lbl = item[2]
        site_id = 0
        for site in sites:
            if site in src:
                site_id = sites.index(site)
                break

        if img_id in img_id_path_dict:
            img_p = img_id_path_dict[img_id]
            dst_path = os.path.join(dst_root, 'clients', str(site_id), lbl)
            shutil.copy2(img_p, dst_path)
            train_count += 1
    print(len(data), train_count)


# 3. Save split dataset
class Ham10kclientsTrainset(Dataset):
    def __init__(self, root, client_id, transform=None):
        super(Ham10kclientsTrainset, self).__init__()
        self.classes = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

        self.root = root
        self.client_id = client_id
        self.transform = transform

        self.images = []
        self.labels = []

        for cls in self.classes:
            cls_p = os.path.join(self.root, 'clients', str(client_id), cls)
            imgs = [f for f in os.listdir(cls_p) if '.jpg' in f]
            for img in imgs:
                img_p = os.path.join(cls_p, img)
                self.images.append(img_p)
                self.labels.append(self.classes.index(cls))

    def __getitem__(self, index):
        img_p = self.images[index]
        label = self.labels[index]
        img = Image.open(img_p).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.images)


class Ham10kclientsTestset(Dataset):
    def __init__(self, root, transform=None):
        super(Ham10kclientsTestset, self).__init__()
        self.classes = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

        self.root = root
        self.transform = transform

        self.images = []
        self.labels = []

        for cls in self.classes:
            cls_p = os.path.join(self.root, 'test', cls)
            imgs = [f for f in os.listdir(cls_p) if '.jpg' in f]
            for img in imgs:
                img_p = os.path.join(cls_p, img)
                self.images.append(img_p)
                self.labels.append(self.classes.index(cls))

    def __getitem__(self, index):
        img_p = self.images[index]
        label = self.labels[index]
        img = Image.open(img_p).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.images)


def save_ham10k_2clients_pt_file():
    root = './dataset/ham10k/ham10k_2clients'
    num_clients = 2

    transform_train = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    transform_test = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    save_root = './dataset/ham10k/ham1k_2clients_pt'
    if os.path.exists(save_root):
        os.system('rm -rf {}'.format(save_root))
    os.mkdir(save_root)

    # Client training set
    for client_id in range(num_clients):
        trainset = Ham10kclientsTrainset(root=root, client_id=client_id, transform=transform_train)
        trainloader = DataLoader(trainset, batch_size=100, shuffle=False, num_workers=4)
        images, labels = None, None
        for idx, data in enumerate(trainloader):
            if idx == 0:
                images, labels = data[0], data[1]
            else:
                images = torch.concat((images, data[0]))
                labels = torch.concat((labels, data[1]))
        print(client_id, images.size(), labels.size())
        torch.save(images, os.path.join(save_root, 'c_{}_train_img.pt'.format(client_id)))
        torch.save(labels, os.path.join(save_root, 'c_{}_train_lbl.pt'.format(client_id)))

    # Testing set
    testset = Ham10kclientsTestset(root=root, transform=transform_test)
    testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)
    images, labels = None, None
    for idx, data in enumerate(testloader):
        if idx == 0:
            images, labels = data[0], data[1]
        else:
            images = torch.concat((images, data[0]))
            labels = torch.concat((labels, data[1]))
    print(images.size(), labels.size())

    torch.save(images, os.path.join(save_root, 'test_img.pt'))
    torch.save(labels, os.path.join(save_root, 'test_lbl.pt'))


##################################################################################################
# Dataset： ham1k_4clients
# 1. Split samples per class referring to nun_test = 23
# 2. Split training set into 4 clients based on the dataset source in csv file
def split_ham10k_4clients():
    ham_root = './dataset/ham10k/'
    ham_data_root = os.path.join(ham_root, 'my_ham_dataset')
    labels = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
    for lbl in labels:
        lbl_p = os.path.join(ham_data_root, lbl)
        imgs = [f for f in os.listdir(lbl_p) if '.jpg' in f]
        print('class: {}, num: {}'.format(lbl, len(imgs)))

    # # Create blank train test folder in ham1k_4clients folder
    dst_root = os.path.join(ham_root, 'ham1k_4clients')
    if os.path.exists(dst_root):
        os.system('rm -rf {}'.format(dst_root))
    os.mkdir(dst_root)

    for fd in ['train', 'test']:
        fd_p = os.path.join(dst_root, fd)
        os.mkdir(fd_p)

    # Copy training images and testing images
    for lbl in labels:
        lbl_p = os.path.join(ham_data_root, lbl)
        imgs = [f for f in os.listdir(lbl_p) if '.jpg' in f]

        for _ in range(10):
            random.shuffle(imgs)

        num_test = 23
        num_train = len(imgs) - num_test

        train_imgs = imgs[: num_train]
        test_imgs = imgs[num_train:]

        # Copy images to train folder
        train_root = os.path.join(dst_root, 'train', lbl)
        if not os.path.exists(train_root):
            os.mkdir(train_root)
        for img in train_imgs:
            img_p = os.path.join(lbl_p, img)
            shutil.copy2(img_p, train_root)

        # Copy images to test folder
        test_root = os.path.join(dst_root, 'test', lbl)
        if not os.path.exists(test_root):
            os.mkdir(test_root)
        for img in test_imgs:
            img_p = os.path.join(lbl_p, img)
            shutil.copy2(img_p, test_root)

    # Check images num in train / test folder
    for lbl in labels:
        lbl_p = os.path.join(ham_data_root, lbl)
        imgs = [f for f in os.listdir(lbl_p) if '.jpg' in f]

        train_lbl_p = os.path.join(dst_root, 'train', lbl)
        train_imgs = [f for f in os.listdir(train_lbl_p) if '.jpg' in f]

        test_lbl_p = os.path.join(dst_root, 'test', lbl)
        test_imgs = [f for f in os.listdir(test_lbl_p) if '.jpg' in f]

        print(len(imgs), len(train_imgs) + len(test_imgs))

    # Split training image and testing images into 4 clients based on the csv file
    sourcs = ['vidir_modern', 'vidir_molemax', 'rosendahl', 'vienna_dias']
    ham_csv_path = os.path.join(ham_root, 'HAM10000_metadata.csv')
    data = []
    with open(ham_csv_path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        next(reader)
        for row in reader:
            data.append(row)
    print(len(data))

    # Create blank clients folder
    client_root = os.path.join(dst_root, 'clients')
    if os.path.exists(client_root):
        os.system('rm -rf {}'.format(client_root))
    os.mkdir(client_root)
    for client_id in range(len(sourcs)):
        client_p = os.path.join(client_root, str(client_id))
        if not os.path.exists(client_p):
            os.mkdir(client_p)

        for lbl in labels:
            lbl_p = os.path.join(client_p, lbl)
            if not os.path.exists(lbl_p):
                os.mkdir(lbl_p)

    # image_id: image_path dict in training set
    img_id_path_dict = {}
    for lbl in labels:
        p = os.path.join(dst_root, 'train', lbl)
        imgs = [f for f in os.listdir(p) if '.jpg' in f]
        for img in imgs:
            img_id = img.replace('.jpg', '').strip()
            img_p = os.path.join(p, img)
            img_id_path_dict[img_id] = img_p

    train_count = 0
    for item in data:
        img_id = item[1]
        src = item[-1]
        lbl = item[2]
        if img_id in img_id_path_dict:
            img_p = img_id_path_dict[img_id]
            dst_path = os.path.join(dst_root, 'clients', str(sourcs.index(src)), lbl)
            shutil.copy2(img_p, dst_path)
            train_count += 1
    print(len(data), train_count)

# 3. Save split dataset
def save_ham10k_4clients_pt_file():
    root = './dataset/ham10k/ham1k_4clients'
    num_clients = 4

    transform_train = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    transform_test = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    save_root = './dataset/ham10k//ham1k_4clients_pt'
    if os.path.exists(save_root):
        os.system('rm -rf {}'.format(save_root))
    os.mkdir(save_root)

    # Client training set
    for client_id in range(num_clients):
        trainset = Ham10kclientsTrainset(root=root, client_id=client_id, transform=transform_train)
        trainloader = DataLoader(trainset, batch_size=100, shuffle=False, num_workers=4)
        images, labels = None, None
        for idx, data in enumerate(trainloader):
            if idx == 0:
                images, labels = data[0], data[1]
            else:
                images = torch.concat((images, data[0]))
                labels = torch.concat((labels, data[1]))
        print(client_id, images.size(), labels.size())
        torch.save(images, os.path.join(save_root, 'c_{}_train_img.pt'.format(client_id)))
        torch.save(labels, os.path.join(save_root, 'c_{}_train_lbl.pt'.format(client_id)))

    # Testing set
    testset = Ham10kclientsTestset(root=root, transform=transform_test)
    testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)
    images, labels = None, None
    for idx, data in enumerate(testloader):
        if idx == 0:
            images, labels = data[0], data[1]
        else:
            images = torch.concat((images, data[0]))
            labels = torch.concat((labels, data[1]))
    print(images.size(), labels.size())

    torch.save(images, os.path.join(save_root, 'test_img.pt'))
    torch.save(labels, os.path.join(save_root, 'test_lbl.pt'))

##################################################################################################
# Dataset：COVID-19 dataset
# 1. Split the training set into non-uniform setting.
def split_covid_nonuniform():
    """
    Split COVID-19 non-uniform dataset from training set.
    oneclass and non-uniform datasets share the same testing set.
    """
    root = './dataset/covid19'
    if os.path.exists(os.path.join(root, 'non-uniform_clients')):
        os.system('rm -rf {}'.format(os.path.join(root, 'non-uniform_clients')))
    os.mkdir(os.path.join(root, 'non-uniform_clients'))
    # label order
    labels = ['Covid-19', 'Normal', 'Lung Opacity', 'Viral', 'Bacterial']

    for lbl in labels:
        lbl_path = os.path.join(root, 'Datasets/5-classes/Train', lbl)
        img_names = os.listdir(lbl_path)
        print(lbl, len(img_names))

    used_imgs = []
    client_count_dict = {
        0: {0: 204, 1: 50, 2: 50, 3: 50, 4: 50},
        1: {0: 50, 1: 204, 2: 50, 3: 50, 4: 50},
        2: {0: 50, 1: 50, 2: 204, 3: 50, 4: 50},
        3: {0: 50, 1: 50, 2: 50, 3: 204, 4: 50},
        4: {0: 50, 1: 50, 2: 50, 3: 50, 4: 204}}

    # Create clients folder
    for client_id in client_count_dict:
        client_root = os.path.join(root, 'non-uniform_clients', 'client{}'.format(client_id))
        if not os.path.exists(client_root):
            os.mkdir(client_root)

        class_count_dict = client_count_dict[client_id]
        for class_id in class_count_dict:
            class_root = os.path.join(root, 'Datasets/5-classes/Train', labels[class_id])
            img_names = [f for f in os.listdir(class_root)]
            count = 0

            client_class_path = os.path.join(client_root, str(class_id))
            if not os.path.exists(client_class_path):
                os.mkdir(client_class_path)

            for img in img_names:
                if img in used_imgs:
                    continue

                img_path = os.path.join(class_root, img)
                shutil.copy2(img_path, client_class_path)
                used_imgs.append(img)
                count += 1

                if count >= class_count_dict[class_id]:
                    print('class {} count: {}'.format(class_id, count))
                    break

        # Check clients image
        client_labels = os.listdir(client_root)
        for lbl in client_labels:
            img_names = [f for f in os.listdir(os.path.join(client_root, lbl))]
            print('client {} class {}: num: {}'.format(client_id, lbl, len(img_names)))

# 2. Split the training set into one-class setting.
def split_1class_covid():
    """
    Split COVID-19 one class dataset.
    """
    root = './dataset/covid19'
    train_root = os.path.join(root, 'Datasets/5-classes/Train')


    # label order
    labels = ['Covid-19', 'Normal', 'Lung_Opacity', 'Viral', 'Bacterial']
    print(labels, len(labels))

    for lbl in labels:
        lbl_path = os.path.join(train_root, lbl)
        img_names = os.listdir(lbl_path)
        print(lbl, len(img_names))

    # Create clients folder
    client_root = os.path.join(root, '1class_clients')
    if os.path.exists(client_root):
        os.system('rm -rf {}'.format(client_root))
    os.mkdir(client_root)

    for lbl in labels:
        client_id = labels.index(lbl)
        client_path = os.path.join(client_root, 'client{}'.format(client_id))
        if os.path.exists(client_path):
            os.system('rm -rf {}'.format(client_path))
        print('copy to {}'.format(client_path))
        os.system('cp -r {} {}'.format(os.path.join(train_root, lbl), os.path.join(client_path)))

# 3. Save the split dataset pt.
class COVIDTestSet(Dataset):
    def __init__(self, root, transform=None):
        super(COVIDTestSet, self).__init__()
        self.images = []
        self.labels = []
        self.transform = transform
        self.root = root
        self.classes = ['Covid-19', 'Normal', 'Lung_Opacity', 'Viral', 'Bacterial']
        print(self.classes)
        for cl in self.classes:
            class_id = self.classes.index(cl)
            img_names = [f for f in os.listdir(os.path.join(self.root, cl))]
            for img in img_names:
                self.images.append(img)
                self.labels.append(class_id)

    def __getitem__(self, index):
        label = self.labels[index]
        img_p = os.path.join(self.root, self.classes[label], self.images[index])

        img = Image.open(img_p).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.images)

def save_covid_test_pt():
    """
    Savg COVID-19 test pt data
    """
    root = './dataset/covid19'
    # Normalize test set same as training set without augmentation
    transform_test = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    testset = COVIDTestSet(os.path.join(root, 'Datasets/5-classes/Test'), transform_test)
    testloader = DataLoader(testset, batch_size=100, num_workers=4)

    images, labels = None, None
    for idx, data in enumerate(testloader):
        if idx == 0:
            images, labels = data[0], data[1]
        else:
            images = torch.concat((images, data[0]))
            labels = torch.concat((labels, data[1]))

    print(images.size(), labels.size())
    if not os.path.exists('./dataset/covid19/covid_1class_5_pt'):
        os.mkdir('./dataset/covid19/covid_1class_5_pt')
    if not os.path.exists('./dataset/covid19/covid_nonuniform_5_pt'):
        os.mkdir('./dataset/covid19/covid_nonuniform_5_pt')

    torch.save(images, os.path.join('./dataset/covid19/covid_1class_5_pt/test_img.pt'))
    torch.save(labels, os.path.join('./dataset/covid19/covid_1class_5_pt/test_lbl.pt'))
    torch.save(images, os.path.join('./dataset/covid19/covid_nonuniform_5_pt/test_img.pt'))
    torch.save(labels, os.path.join('./dataset/covid19/covid_nonuniform_5_pt/test_lbl.pt'))

class COVIDImbTrainSet(Dataset):
    def __init__(self, root, client_id, transform=None):
        super(COVIDImbTrainSet, self).__init__()
        self.images = []
        self.labels = []
        self.transform = transform
        self.client_id = client_id
        self.root = root
        self.num_classes = 5

        for class_id in range(self.num_classes):
            img_names = [f for f in
                         os.listdir(os.path.join(self.root, 'client{}'.format(self.client_id), str(class_id)))]
            for img in img_names:
                self.images.append(img)
                self.labels.append(class_id)

    def __getitem__(self, index):
        label = self.labels[index]
        img_p = os.path.join(self.root, 'client{}'.format(self.client_id), str(label),
                             self.images[index])

        img = Image.open(img_p).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.images)

def save_covid_nonuniform_pt():
    """
    Save COVID-19 non-uniform pt data
    """
    root = './dataset/covid19/non-uniform_clients'
    client_ids = list(range(5))
    transform_train = transforms.Compose([
        transforms.Resize([224, 224]),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    for client_id in client_ids:

        trainset = COVIDImbTrainSet(root, client_id, transform_train)
        trainloader = DataLoader(trainset, batch_size=100, num_workers=4)

        images, labels = None, None
        for idx, data in enumerate(trainloader):
            if idx == 0:
                images, labels = data[0], data[1]
            else:
                images = torch.concat((images, data[0]))
                labels = torch.concat((labels, data[1]))

        print(images.size(), labels.size())

        torch.save(images, os.path.join('./dataset/covid19/covid_nonuniform_5_pt/c_{}_train_img.pt'.format(client_id)))
        torch.save(labels, os.path.join('./dataset/covid19/covid_nonuniform_5_pt/c_{}_train_lbl.pt'.format(client_id)))

class COVIDOneClassTrainSet(Dataset):
    def __init__(self, root, client_id, transform=None):
        super(COVIDOneClassTrainSet, self).__init__()
        self.images = []
        self.labels = []
        self.transform = transform
        self.client_id = client_id
        self.root = root
        self.images = [f for f in os.listdir(os.path.join(root,
                                                          'client{}'.format(client_id)))]
        self.labels = [int(client_id) for _ in range(len(self.images))]

    def __getitem__(self, index):
        img_p = os.path.join(self.root, 'client{}'.format(self.client_id), self.images[index])
        label = self.labels[index]
        img = Image.open(img_p).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.images)

def save_covid_oneclass_pt():
    """
    Save COVID oneclass pt data
    """
    root = './dataset/covid19/1class_clients'
    client_ids = list(range(5))
    transform_train = transforms.Compose([
        transforms.Resize([224, 224]),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    for client_id in client_ids:

        trainset = COVIDOneClassTrainSet(root, client_id, transform_train)
        trainloader = DataLoader(trainset, batch_size=100, num_workers=4)

        images, labels = None, None
        for idx, data in enumerate(trainloader):
            if idx == 0:
                images, labels = data[0], data[1]
            else:
                images = torch.concat((images, data[0]))
                labels = torch.concat((labels, data[1]))

        print(images.size(), labels.size())

        torch.save(images, os.path.join('./dataset/covid19/covid_1class_5_pt/c_{}_train_img.pt'.format(client_id)))
        torch.save(labels, os.path.join('./dataset/covid19/covid_1class_5_pt/c_{}_train_lbl.pt'.format(client_id)))

##################################################################################################
# Dataset：PBC dataset
# 1. Split the training set into one-class setting.
def split_pbc_1_class():
    """
    Split PGC one class set
    """
    root = './dataset/pbc/PBC_dataset_normal_DIB/PBC_dataset_normal_DIB'
    save_root = './dataset/pbc/'
    # label order
    labels = ['eosinophil', 'erythroblast', 'ig', 'lymphocyte', 'basophil', 'monocyte',
              'neutrophil', 'platelet']
    for lbl in labels:
        lbl_path = os.path.join(root, lbl)
        img_names = os.listdir(lbl_path)
        print(lbl, len(img_names))

    # Create training, and testing set by randomly sampling 243 images from original images
    # create empty train and test folders
    train_root = os.path.join(save_root, 'train')
    if os.path.exists(train_root):
        os.system('rm -rf {}'.format(train_root))
    os.mkdir(train_root)

    test_root = os.path.join(save_root, 'test')
    if os.path.exists(test_root):
        os.system('rm -rf {}'.format(test_root))
    os.mkdir(test_root)

    for lbl in labels:
        lbl_path = os.path.join(root, lbl)
        img_names = [f for f in os.listdir(lbl_path) if '.jpg' in f]
        print(lbl, len(img_names))
        for _ in range(10):
            random.shuffle(img_names)
        test_imgs = img_names[: 243]
        train_imgs = img_names[243:]

        # copy images to train / test folder
        train_class_root = os.path.join(train_root, lbl)
        test_class_root = os.path.join(test_root, lbl)

        if not os.path.exists(train_class_root):
            os.mkdir(train_class_root)
        if not os.path.exists(test_class_root):
            os.mkdir(test_class_root)

        for img in train_imgs:
            shutil.copy2(os.path.join(lbl_path, img), train_class_root)
        for img in test_imgs:
            shutil.copy2(os.path.join(lbl_path, img), test_class_root)

        # Check images num
        train_img_names = [f for f in os.listdir(train_class_root) if '.jpg' in f]
        test_img_names = [f for f in os.listdir(test_class_root) if '.jpg' in f]
        print(len(img_names), len(train_img_names), len(test_img_names),
              len(train_img_names) + len(test_img_names))

        # Create clients folder
    client_root = os.path.join(save_root, '1class_clients')
    if os.path.exists(client_root):
        os.system('rm -rf {}'.format(client_root))
    os.mkdir(client_root)

    for lbl in labels:
        client_id = labels.index(lbl)
        client_path = os.path.join(client_root, 'client{}'.format(client_id))
        if os.path.exists(client_path):
            os.system('rm -rf {}'.format(client_path))
        print('copy to {}'.format(client_path))
        os.system('cp -r {} {}'.format(os.path.join(train_root, lbl), os.path.join(client_path)))

# 2. Split the training set into non-uniform setting.
def split_pbc_nonuniform_class():
    """
    Splt PBC imbalance dataset from training set.
    oneclass and imb datasets share the same testing set.
    """
    root = './dataset/pbc'
    if os.path.exists(os.path.join(root, 'nonuniform_clients')):
        os.system('rm -rf {}'.format(os.path.join(root, 'nonuniform_clients')))
    os.mkdir(os.path.join(root, 'nonuniform_clients'))
    # label order
    labels = ['eosinophil', 'erythroblast', 'ig', 'lymphocyte', 'basophil',
              'monocyte', 'neutrophil', 'platelet']

    for lbl in labels:
        lbl_path = os.path.join(root, 'train', lbl)
        img_names = os.listdir(lbl_path)
        print(lbl, len(img_names))

    used_imgs = []
    client_count_dict = {
        0: {0: 1439, 1: 93, 2: 189, 3: 69, 4: 69, 5: 84, 6: 221, 7: 150},
        1: {0: 205, 1: 657, 2: 189, 3: 69, 4: 69, 5: 84, 6: 221, 7: 150},
        2: {0: 205, 1: 93, 2: 1329, 3: 69, 4: 69, 5: 84, 6: 221, 7: 150},
        3: {0: 205, 1: 93, 2: 189, 3: 488, 4: 69, 5: 84, 6: 221, 7: 150},
        4: {0: 205, 1: 93, 2: 189, 3: 69, 4: 492, 5: 84, 6: 221, 7: 150},
        5: {0: 205, 1: 93, 2: 189, 3: 69, 4: 69, 5: 589, 6: 221, 7: 150},
        6: {0: 205, 1: 93, 2: 189, 3: 69, 4: 69, 5: 84, 6: 1539, 7: 150},
        7: {0: 205, 1: 93, 2: 189, 3: 69, 4: 69, 5: 84, 6: 221, 7: 1055}}

    # Create clients folder
    for client_id in client_count_dict:
        client_root = os.path.join(root, 'nonuniform_clients', 'client{}'.format(client_id))
        if not os.path.exists(client_root):
            os.mkdir(client_root)

        class_count_dict = client_count_dict[client_id]
        for class_id in class_count_dict:
            class_root = os.path.join(root, 'train', labels[class_id])
            img_names = [f for f in os.listdir(class_root) if '.jpg' in f]
            count = 0

            client_class_path = os.path.join(client_root, str(class_id))
            if not os.path.exists(client_class_path):
                os.mkdir(client_class_path)

            for img in img_names:
                if img in used_imgs:
                    continue

                img_path = os.path.join(class_root, img)
                shutil.copy2(img_path, client_class_path)
                used_imgs.append(img)
                count += 1

                if count >= class_count_dict[class_id]:
                    print('class {} count: {}'.format(class_id, count))
                    break

        # Check clients image
        client_labels = os.listdir(client_root)
        for lbl in client_labels:
            img_names = [f for f in os.listdir(os.path.join(client_root, lbl)) if '.jpg' in f]
            print('client {} class {}: num: {}'.format(client_id, lbl, len(img_names)))


# 3. Save the split dataset pt.
class PBCTestSet(Dataset):
    def __init__(self, root, transform=None):
        super(PBCTestSet, self).__init__()
        self.images = []
        self.labels = []
        self.transform = transform
        self.root = root
        self.classes = ['eosinophil', 'erythroblast', 'ig', 'lymphocyte', 'basophil',
                        'monocyte', 'neutrophil', 'platelet']
        print(self.classes)
        for cl in self.classes:
            class_id = self.classes.index(cl)
            img_names = [f for f in os.listdir(os.path.join(self.root, cl)) if '.jpg' in f]
            for img in img_names:
                self.images.append(img)
                self.labels.append(class_id)

    def __getitem__(self, index):
        label = self.labels[index]
        img_p = os.path.join(self.root, self.classes[label], self.images[index])

        img = Image.open(img_p).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.images)

def save_pbc_test_pt():
    """
    Save PBC test pt data
    """
    root = './dataset/pbc'
    # Normalize test set same as training set without augmentation
    transform_test = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    testset = PBCTestSet(os.path.join(root, 'test'), transform_test)
    testloader = DataLoader(testset, batch_size=100, num_workers=4)

    images, labels = None, None
    for idx, data in enumerate(testloader):
        if idx == 0:
            images, labels = data[0], data[1]
        else:
            images = torch.concat((images, data[0]))
            labels = torch.concat((labels, data[1]))

    print(images.size(), labels.size())

    if not os.path.exists('./dataset/pbc/pbc_1class_8_pt'):
        os.mkdir('./dataset/pbc/pbc_1class_8_pt')

    if not os.path.exists('./dataset/pbc/pbc_nonuniform_8_pt'):
        os.mkdir('./dataset/pbc/pbc_nonuniform_8_pt')

    torch.save(images, os.path.join('./dataset/pbc/pbc_1class_8_pt/test_img.pt'))
    torch.save(labels, os.path.join('./dataset/pbc/pbc_1class_8_pt/test_lbl.pt'))
    torch.save(images, os.path.join('./dataset/pbc/pbc_nonuniform_8_pt/test_img.pt'))
    torch.save(labels, os.path.join('./dataset/pbc/pbc_nonuniform_8_pt/test_lbl.pt'))


class PBCOneClassTrainSet(Dataset):
    def __init__(self, root, client_id, transform=None):
        super(PBCOneClassTrainSet, self).__init__()
        self.images = []
        self.labels = []
        self.transform = transform
        self.client_id = client_id
        self.root = root
        self.images = [f for f in os.listdir(os.path.join(root,
                                                          'client{}'.format(client_id))) if '.jpg' in f]
        self.labels = [int(client_id) for _ in range(len(self.images))]

    def __getitem__(self, index):
        img_p = os.path.join(self.root, 'client{}'.format(self.client_id), self.images[index])
        label = self.labels[index]
        img = Image.open(img_p).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.images)

def save_pbc_oneclass_pt():
    """
    Save PBC oneclass pt data
    """
    root = './dataset/pbc/1class_clients'
    client_ids = list(range(8))
    transform_train = transforms.Compose([
        transforms.Resize([224, 224]),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    for client_id in client_ids:

        trainset = PBCOneClassTrainSet(root, client_id, transform_train)
        trainloader = DataLoader(trainset, batch_size=100, num_workers=4)

        images, labels = None, None
        for idx, data in enumerate(trainloader):
            if idx == 0:
                images, labels = data[0], data[1]
            else:
                images = torch.concat((images, data[0]))
                labels = torch.concat((labels, data[1]))

        print(images.size(), labels.size())

        torch.save(images, os.path.join('./dataset/pbc/pbc_1class_8_pt/c_{}_train_img.pt'.format(client_id)))
        torch.save(labels, os.path.join('./dataset/pbc/pbc_1class_8_pt/c_{}_train_lbl.pt'.format(client_id)))


class PBCImbTrainSet(Dataset):
    def __init__(self, root, client_id, transform=None):
        super(PBCImbTrainSet, self).__init__()
        self.images = []
        self.labels = []
        self.transform = transform
        self.client_id = client_id
        self.root = root
        self.num_classes = 8

        for class_id in range(self.num_classes):
            img_names = [f for f in
                         os.listdir(os.path.join(self.root, 'client{}'.format(self.client_id), str(class_id)))
                         if '.jpg' in f]
            for img in img_names:
                self.images.append(img)
                self.labels.append(class_id)

    def __getitem__(self, index):
        label = self.labels[index]
        img_p = os.path.join(self.root, 'client{}'.format(self.client_id), str(label),
                             self.images[index])

        img = Image.open(img_p).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.images)

def save_pbc_nonuniform_pt():
    """
    Save PBC imb pt data
    """
    root = './dataset/pbc/nonuniform_clients'
    client_ids = list(range(8))
    transform_train = transforms.Compose([
        transforms.Resize([224, 224]),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    for client_id in client_ids:

        trainset = PBCImbTrainSet(root, client_id, transform_train)
        trainloader = DataLoader(trainset, batch_size=100, num_workers=4)

        images, labels = None, None
        for idx, data in enumerate(trainloader):
            if idx == 0:
                images, labels = data[0], data[1]
            else:
                images = torch.concat((images, data[0]))
                labels = torch.concat((labels, data[1]))

        print(images.size(), labels.size())

        torch.save(images, os.path.join('./dataset/pbc/pbc_nonuniform_8_pt/c_{}_train_img.pt'.format(client_id)))
        torch.save(labels, os.path.join('./dataset/pbc/pbc_nonuniform_8_pt/c_{}_train_lbl.pt'.format(client_id)))

if __name__ == '__main__':
    # save_ham10k_dataset()
    # split_ham10k_2clients()
    # save_ham10k_2clients_pt_file()
    # split_ham10k_4clients()
    # save_ham10k_4clients_pt_file()
    ############################################
    # split_covid_nonuniform()
    # split_1class_covid()
    # save_covid_test_pt()
    # save_covid_nonuniform_pt()
    # save_covid_oneclass_pt()
    ############################################
    # split_pbc_1_class()
    # save_pbc_test_pt()
    # save_pbc_oneclass_pt()
    # split_pbc_nonuniform_class()
    save_pbc_nonuniform_pt()