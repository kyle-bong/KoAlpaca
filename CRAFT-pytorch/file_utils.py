# -*- coding: utf-8 -*-
import os
import numpy as np
import cv2
import imgproc
import pytesseract
from imutils.perspective import four_point_transform
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, AutoTokenizer
import unicodedata

processor = TrOCRProcessor.from_pretrained("ddobokki/ko-trocr")
model = VisionEncoderDecoderModel.from_pretrained("ddobokki/ko-trocr")
tokenizer = AutoTokenizer.from_pretrained("ddobokki/ko-trocr")

# borrowed from https://github.com/lengstrom/fast-style-transfer/blob/master/src/utils.py
def get_files(img_dir):
    imgs, masks, xmls = list_files(img_dir)
    return imgs, masks, xmls

def list_files(in_path):
    img_files = []
    mask_files = []
    gt_files = []
    for (dirpath, dirnames, filenames) in os.walk(in_path):
        for file in filenames:
            filename, ext = os.path.splitext(file)
            ext = str.lower(ext)
            if ext == '.jpg' or ext == '.jpeg' or ext == '.gif' or ext == '.png' or ext == '.pgm':
                img_files.append(os.path.join(dirpath, file))
            elif ext == '.bmp':
                mask_files.append(os.path.join(dirpath, file))
            elif ext == '.xml' or ext == '.gt' or ext == '.txt':
                gt_files.append(os.path.join(dirpath, file))
            elif ext == '.zip':
                continue
    # img_files.sort()
    # mask_files.sort()
    # gt_files.sort()
    return img_files, mask_files, gt_files

def saveResult(img_file, img, boxes, dirname='./result/', verticals=None, texts=None):
        """ save text detection result one by one
        Args:
            img_file (str): image file name
            img (array): raw image context
            boxes (array): array of result file
                Shape: [num_detections, 4] for BB output / [num_detections, 4] for QUAD output
        Return:
            None
        """
        img = np.array(img)
        
        # make result file list
        filename, file_ext = os.path.splitext(os.path.basename(img_file))

        # result directory
        res_file = dirname + "res_" + filename + '.txt'
        res_img_file = dirname + "res_" + filename + '.jpg'
        
        # 전체 OCR
        # OCR
        pixel_values = processor(img, return_tensors='pt').pixel_values
        generated_ids = model.generate(pixel_values)
        generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        generated_text = unicodedata.normalize("NFC", generated_text)
        
        with open(res_img_file + 'total_text.txt', 'w') as f:
            f.write(generated_text)
            
        if not os.path.isdir(dirname):
            os.mkdir(dirname)

        with open(res_file, 'w') as f:
            ocr_results = []
            for i, box in enumerate(boxes):
                poly = np.array(box).astype(np.int32).reshape((-1))
                strResult = ','.join([str(p) for p in poly]) + '\r\n'
                f.write(strResult)

                poly = poly.reshape(-1, 2)
                cv2.polylines(img, [poly.reshape((-1, 1, 2))], True, color=(0, 0, 255), thickness=2)
                
                # Poly line 내 이미지만 따로 저장 및 OCR
                # region_mask = np.zeros_like(img)
                # cv2.fillPoly(region_mask, [poly.reshape((-1, 1, 2))], color=(255, 255, 255))
                # region_img = cv2.bitwise_and(img, region_mask)
                x, y, w, h = cv2.boundingRect(poly)
                region_img = img[y:y+h, x:x+w]
                
                # OCR
                pixel_values = processor(region_img, return_tensors='pt').pixel_values
                generated_ids = model.generate(pixel_values)
                generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                generated_text = unicodedata.normalize("NFC", generated_text)

                ocr_results.append(str(generated_text))
                
#                 cv2.imwrite(res_img_file + '_' + str(i) + '_poly.jpg', region_img)
                
                ptColor = (0, 255, 255)
                if verticals is not None:
                    if verticals[i]:
                        ptColor = (255, 0, 0)

                if texts is not None:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.5
                    cv2.putText(img, "{}".format(texts[i]), (poly[0][0]+1, poly[0][1]+1), font, font_scale, (0, 0, 0), thickness=1)
                    cv2.putText(img, "{}".format(texts[i]), tuple(poly[0]), font, font_scale, (0, 255, 255), thickness=1)

        # Save result image
        cv2.imwrite(res_img_file, img)
        
        # save ocr result
        with open(res_img_file + '_text.txt', 'w') as f:
            f.write(' '.join(ocr_results))

