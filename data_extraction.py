import cv2
import numpy as np

# Extract images for training and testing
ANNOTATION_FILE_LOCATION = "G:/Project/celebA/Anno/list_bbox_celeba.txt"
annotation_file = open(ANNOTATION_FILE_LOCATION, "r")

IMAGE_LOCATION = "G:/Project/celebA/Img/img_align_celeba/"
IMAGE_FORMAT = ".jpg"

###########################################
#### Saving faces for training dataset ####
###########################################
count = 1
FACES_TRAINING_SRC_LOCATION = "G:/Project/training/faces/faces-"

faces_count = 1
for line in annotation_file:
    if count <= 2:
        count += 1
        continue

    count += 1
    line = line.rstrip()
    break_line = line.split()
    file_name = break_line[0]

    image_src = IMAGE_LOCATION + file_name
    print("Processing ... " + image_src)
    img = cv2.imread(image_src, cv2.IMREAD_COLOR)
    result = img
    height, width = result.shape[:2]
    max_height = 60
    max_width = 60

    try:
        if max_height < height or max_width < width:
            # get scaling factor
            scaling_factor = max_height / float(height)
            if max_width / float(width) < scaling_factor:
                scaling_factor = max_width / float(width)
            
            # resize image
            result = cv2.resize(result, (60, 60), interpolation=cv2.INTER_AREA)
    except cv2.error as e:
        print("Skipped Image: " + image_src)
        continue
    
    # cv2.imshow("resized", result)
    faces_file_name = FACES_TRAINING_SRC_LOCATION + str(faces_count) + IMAGE_FORMAT
    print(faces_file_name)
    cv2.imwrite(faces_file_name, result)
    print("Saving ... " + faces_file_name)
    faces_count += 1

    if faces_count == 10001:
        break


#######################################
#### Saving faces for test dataset ####
#######################################
count = 1
FACES_TEST_SRC_LOCATION = "G:/Project/test/faces/faces-"

faces_count = 1
for line in annotation_file:
    if count <= 50002:
        count += 1
        continue

    count += 1
    line = line.rstrip()
    break_line = line.split()
    file_name = break_line[0]

    image_src = IMAGE_LOCATION + file_name
    print("Processing ... " + image_src)
    img = cv2.imread(image_src, cv2.IMREAD_COLOR)
    result = img
    height, width = result.shape[:2]
    max_height = 60
    max_width = 60

    try:
        if max_height < height or max_width < width:
            # get scaling factor
            scaling_factor = max_height / float(height)
            if max_width / float(width) < scaling_factor:
                scaling_factor = max_width / float(width)
            
            # resize image
            result = cv2.resize(result, (60, 60), interpolation=cv2.INTER_AREA)
    except cv2.error as e:
        print("Skipped Image: " + image_src)
        continue
    
    # cv2.imshow("resized", result)
    faces_file_name = FACES_TEST_SRC_LOCATION + str(faces_count) + IMAGE_FORMAT
    print(faces_file_name)
    cv2.imwrite(faces_file_name, result)
    print("Saving ... " + faces_file_name)
    faces_count += 1

    if faces_count == 1001:
        break

IMAGE_NON_CROPPED_LOCATION = "G:/Project/celebA/Img/img_celeba/"
IMAGE_FORMAT = ".jpg"

###############################################
#### Saving non-faces for training dataset ####
###############################################
count = 1
NON_FACES_TRAINING_SRC_LOCATION = "G:/Project/training/non_faces/non_faces-"

non_faces_count = 1
for line in annotation_file:
    if count <= 2:
        count += 1
        continue

    count += 1
    line = line.rstrip()
    break_line = line.split()
    file_name = break_line[0]

    image_src = IMAGE_LOCATION + file_name
    print("Processing ... " + image_src)
    img = cv2.imread(image_src, cv2.IMREAD_COLOR)
    resize_img = img[0: 60, 0 : 60]
    non_faces_file_name = NON_FACES_TRAINING_SRC_LOCATION + str(non_faces_count) + IMAGE_FORMAT
    cv2.imwrite(non_faces_file_name, resize_img)
    print("Saving ... " + non_faces_file_name)
    non_faces_count += 1

    if non_faces_count == 10001:
        break

###########################################
#### Saving non-faces for test dataset ####
###########################################
count = 1
NON_FACES_TEST_SRC_LOCATION = "G:/Project/test/non_faces/non_faces-"

non_faces_count = 1
for line in annotation_file:
    if count <= 50002:
        count += 1
        continue

    count += 1
    line = line.rstrip()
    break_line = line.split()
    file_name = break_line[0]

    image_src = IMAGE_LOCATION + file_name
    print("Processing ... " + image_src)
    img = cv2.imread(image_src, cv2.IMREAD_COLOR)
    resize_img = img[0: 60, 0 : 60]
    non_faces_file_name = NON_FACES_TEST_SRC_LOCATION + str(non_faces_count) + IMAGE_FORMAT
    cv2.imwrite(non_faces_file_name, resize_img)
    print("Saving ... " + non_faces_file_name)
    non_faces_count += 1

    if non_faces_count == 1001:
        break