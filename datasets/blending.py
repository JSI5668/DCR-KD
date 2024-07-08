from PIL import Image
import numpy as np
import cv2
import random
import matplotlib.pylab as plt
from glob import glob
import os

def load_image(name):
    img = Image.open(name)
    return np.array(img)



def random_new_label_gray(name):

    img = Image.open(name)

    px = img.load()

    r_array = np.asarray(img)
    px_array = np.asarray(px)

    full_pixel = []

    a1 = random.choices(range(0, 256), k=3)
    a2 = random.choices(range(0, 256), k=3)
    a3 = random.choices(range(0, 256), k=3)
    a4 = random.choices(range(0, 256), k=3)
    a5 = random.choices(range(0, 256), k=3)
    a6 = random.choices(range(0, 256), k=3)
    a7 = random.choices(range(0, 256), k=3)
    a8 = random.choices(range(0, 256), k=3)
    a9 = random.choices(range(0, 256), k=3)
    a10 = random.choices(range(0, 256), k=3)
    a11 = random.choices(range(0, 256), k=3)
    a12 = random.choices(range(0, 256), k=3)

    for i in range(0, 240):
        void_list = []
        for j in range(0, 320):
            try:
                rgb = r_array[i][j]
            except:
                print(i)
                print(j)

            # print(rgb)
            # rgb = r.getpixel((i, j))
            # print(rgb)
            if rgb == 0: ##(128,128,128) sky
                # px[i, j] = (a1[0], a1[1], a1[2])
                void_list.append([a1[0], a1[1], a1[2]])

            elif rgb == 1: ##(128, 0, 0) Building
                # px[i, j] = (a2[0], a2[1], a2[2])
                void_list.append([a2[0], a2[1], a2[2]])

            elif rgb == 2: ##(192, 192, 128) Pole
                void_list.append([a3[0], a3[1], a3[2]])

            elif rgb == 3: ##(128, 64, 128) Road
                void_list.append([a4[0], a4[1], a4[2]])

            elif rgb == 4: ##(0, 0, 192) Sidewalk
                void_list.append([a5[0], a5[1], a5[2]])

            elif rgb == 5: ##(128, 128, 0) Tree
                void_list.append([a6[0], a6[1], a6[2]])

            elif rgb == (6):  ##(192, 128, 128) SignSymbole
                void_list.append([a7[0], a7[1], a7[2]])

            elif rgb == (7):   ##(64, 64, 128) Fence
                void_list.append([a8[0], a8[1], a8[2]])

            elif rgb == (8):  ##(64, 0, 128) Car
                void_list.append([a9[0], a9[1], a9[2]])

            elif rgb == (9):  ##(64, 64, 0) Pedestrian
                void_list.append([a10[0], a10[1], a10[2]])

            elif rgb == (10):  ##(0, 128, 192) Bicyclist
                void_list.append([a11[0], a11[1], a11[2]])

            else:
                # void_list.append([a12[0], a12[1], a12[2]])
                void_list.append([0, 0, 0])

            # print(rgb)
        full_pixel.append(void_list)
    px_np = np.array(full_pixel)

    px_np = px_np.astype(np.uint8)

    # plt.imshow(px_np)
    # plt.show()

    # plt.imsave('C:/Users/JSIISPR/Desktop/github_deeplab/amazing/Amazing-Semantic-Segmentation-master/newcolor2/alpha_labels_Gray_blackno_0.40/{0}.png'.format(idx), px_np )

    # px_np.save('C:/Users/JSIISPR/Desktop/github_deeplab/amazing/Amazing-Semantic-Segmentation-master/newcolor2/labels2/{0}.png'.format(idx))
    # return np.array(px_np)
    return px_np




def alpha_blending(image, label_2, idx, epoch):


    alpha = np.random.uniform(0.70, 0.99)
    # alpha = 0.40

    # image = cv2.resize(image, (256,256), cv2.INTER_LINEAR)

    blended = (label_2 * alpha) + (image * (1-alpha))              # 방식1
    blended = blended.astype(np.uint8) # 소수점 제거

    # blended = cv2.resize(blended, (320,240), cv2.INTER_LINEAR)
    path = 'D:/Dataset/blending_2/{0}'.format(epoch)
    os.makedirs(path, exist_ok=True)
    cv2.imwrite(os.path.join(path,'{0}.png'.format(idx)), blended)
    #
    return np.array(blended)

