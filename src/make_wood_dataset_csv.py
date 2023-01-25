from cgi import test
import os 
import csv 
import parser
from argparse import ArgumentParser
from collections import defaultdict


def argparser():
    parser = ArgumentParser()
    parser.add_argument('--train_folder_path',
                        type=str,
                        default='../wood_dataset/train')
    parser.add_argument('--valid_folder_path',
                        type=str,
                        default='../wood_dataset/val')
    parser.add_argument('--test_folder_path',
                        type=str,
                        default='../wood_dataset/test')
    return parser


def make_wood_dataset_csv(train_folder_path, valid_folder_path, test_folder_path):
    # 学習に関するデータのcsvを作成
    image_folder_path = os.path.join(train_folder_path, 'images')
    image_filename_list = os.listdir(image_folder_path)
    train_rows = list()
    for image_filename in image_filename_list:
        image_path = os.path.join(image_folder_path, image_filename)
        label = 0  # 学習は正常画像のみなので、labelは0
        train_rows.append([image_path, label, None])
        
    with open(os.path.join(train_folder_path, 'train_labels.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['image_path', 'label', 'mask_path'])
        writer.writerows(train_rows)
        
    # 検証に関するデータのcsvを作成
    image_folder_path = os.path.join(valid_folder_path, 'images')
    image_filename_list = os.listdir(image_folder_path)
    valid_rows = list()
    for image_filename in image_filename_list:
        image_path = os.path.join(image_folder_path, image_filename)
        label = 0  # 学習は正常画像のみなので、labelは0
        valid_rows.append([image_path, label, None])
        
    with open(os.path.join(valid_folder_path, 'valid_labels.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['image_path', 'label', 'mask_path'])
        writer.writerows(valid_rows)
    
    # 推論に関するデータのcsv作成
    image_folder_path = os.path.join(test_folder_path, 'images')
    image_filename_list = os.listdir(image_folder_path)
    AND_mask_folder_path = os.path.join(test_folder_path, 'masks', 'AND')
    AND_mask_filename_list = os.listdir(AND_mask_folder_path)
    XOR_mask_folder_path = os.path.join(test_folder_path, 'masks', 'XOR')
    XOR_mask_filename_list = os.listdir(XOR_mask_folder_path)
    test_AND_rows = list()
    test_XOR_rows = list()
    for image_filename in image_filename_list:
        image_path = os.path.join(image_folder_path, image_filename)
        AND_label = 0
        AND_mask_path = None
        if image_filename in AND_mask_filename_list:
            AND_label = 1
            AND_mask_path = os.path.join(AND_mask_folder_path, image_filename)
        test_AND_rows.append([image_path, AND_label, AND_mask_path])
        
        XOR_label = 0
        XOR_mask_path = None
        if image_filename in XOR_mask_filename_list:
            XOR_label = 1
            XOR_mask_path = os.path.join(XOR_mask_folder_path, image_filename)
        test_XOR_rows.append([image_path, XOR_label, XOR_mask_path])
    
    with open(os.path.join(test_folder_path, 'test_AND_labels.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['image_path', 'label', 'mask_path'])
        writer.writerows(test_AND_rows)
    with open(os.path.join(test_folder_path, 'test_XOR_labels.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['image_path', 'label', 'mask_path'])
        writer.writerows(test_XOR_rows)


if __name__=='__main__':
    parser = argparser()
    args = parser.parse_args()
    
    make_wood_dataset_csv(train_folder_path=args.train_folder_path, valid_folder_path=args.valid_folder_path, test_folder_path=args.test_folder_path)
        
        