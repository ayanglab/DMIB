'''
input:
datapath_3D
ct_list
savepath_2Dmontage
'''
# parameters for
import numpy as np
import pandas as pd
import SimpleITK as sitk
import sys
sys.path.append("..")
from image_processing import image_3D_normalisation
import copy
import cv2
import os
import matplotlib.pyplot as plt
import random
from utils.train import setup_seed
import os, glob, json

def transform_label_to_3D(label):
    img = np.zeros([label.shape[0],label.shape[1], 3])
    for i in range(label.shape[0]):
        for j in range(label.shape[1]):
            if label[i,j] == 1: # lung: red
                img[i, j, 0] = 1
            elif label[i,j] == 2: # ggo: green
                img[i, j, 1] = 1
            elif label[i,j] == 3:
                img[i, j, 2] = 1
    return img


def image_compose_multiple_clockwise(image, mask, x=2, z=2, to_resize=False, image_resize=[350,350], start_loc=0, end_loc=0):


    [sz, sx, sy] = image.shape
    combine_z = np.zeros([x * image_resize[0], z * image_resize[1]])
    combine_z_mask = np.zeros([x * image_resize[0], z * image_resize[1], 3])

    # if start_loc == 0:
    #     start_loc = start_loc+1
    num_interval = int((end_loc - start_loc)/(x*z))
    L = []
    for index in range(x * z):
        L.append(random.sample(range(start_loc+index * num_interval, start_loc + (index+1) * num_interval), 1)[0]) # i

    if not to_resize:
        # clockwise
        combine_z[0 * sx:(0 + 1) * sx, 0 * sy:(0 + 1) * sy] = image[L[0], :, ::]
        combine_z[0 * sx:(0 + 1) * sx, 1 * sy:(1 + 1) * sy] = image[L[1], :, ::]
        combine_z[1 * sx:(1 + 1) * sx, 0 * sy:(0 + 1) * sy] = image[L[3], :, ::] # clockwise
        combine_z[1 * sx:(1 + 1) * sx, 1 * sy:(1 + 1) * sy] = image[L[2], :, ::]

        combine_z_mask[0 * sx:(0 + 1) * sx, 0 * sy:(0 + 1) * sy, :] = transform_label_to_3D(mask[L[0], :, ::])
        combine_z_mask[0 * sx:(0 + 1) * sx, 1 * sy:(1 + 1) * sy, :] = transform_label_to_3D(mask[L[1], :, ::])
        combine_z_mask[1 * sx:(1 + 1) * sx, 0 * sy:(0 + 1) * sy, :] = transform_label_to_3D(mask[L[3], :, ::])
        combine_z_mask[1 * sx:(1 + 1) * sx, 1 * sy:(1 + 1) * sy, :] = transform_label_to_3D(mask[L[2], :, ::])
    else:
        blank=0
    return combine_z, combine_z_mask

def image_compose_multiple_clockwise_two(image, image_masked, mask, x=2, z=2, to_resize=False, image_resize=[350,350], start_loc=0, end_loc=0):


    [sz, sx, sy] = image.shape
    combine_z = np.zeros([x * image_resize[0], z * image_resize[1]])
    combine_z_masked = np.zeros([x * image_resize[0], z * image_resize[1]])
    combine_z_mask = np.zeros([x * image_resize[0], z * image_resize[1], 3])

    # if start_loc == 0:
    #     start_loc = start_loc+1
    num_interval = int((end_loc - start_loc)/(x*z))
    L = []
    for index in range(x * z):
        L.append(random.sample(range(start_loc+index * num_interval, start_loc + (index+1) * num_interval), 1)[0]) # i

    if not to_resize:
        # clockwise
        combine_z[0 * sx:(0 + 1) * sx, 0 * sy:(0 + 1) * sy] = image[L[0], :, ::]
        combine_z[0 * sx:(0 + 1) * sx, 1 * sy:(1 + 1) * sy] = image[L[1], :, ::]
        combine_z[1 * sx:(1 + 1) * sx, 0 * sy:(0 + 1) * sy] = image[L[3], :, ::] # clockwise
        combine_z[1 * sx:(1 + 1) * sx, 1 * sy:(1 + 1) * sy] = image[L[2], :, ::]

        combine_z_masked[0 * sx:(0 + 1) * sx, 0 * sy:(0 + 1) * sy] = image_masked[L[0], :, ::]
        combine_z_masked[0 * sx:(0 + 1) * sx, 1 * sy:(1 + 1) * sy] = image_masked[L[1], :, ::]
        combine_z_masked[1 * sx:(1 + 1) * sx, 0 * sy:(0 + 1) * sy] = image_masked[L[3], :, ::]  # clockwise
        combine_z_masked[1 * sx:(1 + 1) * sx, 1 * sy:(1 + 1) * sy] = image_masked[L[2], :, ::]

    else:
        blank=0
    return combine_z, combine_z_masked


def image_3Dto2D(np_ct_img, np_mask, np_mask_binary, np_leision1_binary, np_leision2_binary, ct_name, ct_mask_name, crop_min_max = [-1024, 350], to_resize=False, image_resize = [350,350],  num_2D=[2,2], if_save = True):

    sum_pixel = np_mask_binary.shape[1] * np_mask_binary.shape[2]
    plane_select = [t for t in range(np_mask_binary.shape[0]) if
                    np_mask_binary[t, :, :].sum() > 0 ]
    if False:
        # plot lung and lesion
        lung = [np_mask_binary[t, :, :].sum() for t in plane_select]
        lesion1 = [np_leision1_binary[t, :, :].sum() for t in plane_select]
        lesion2 = [np_leision2_binary[t, :, :].sum() for t in plane_select]

        x_axis = np.linspace(0,1,len(lung))
        plt.plot(x_axis, lung, color='green', linewidth=1)
        plt.plot(x_axis, lesion1, color='red', linewidth=1)
        plt.plot(x_axis, lesion2, color='blue', linewidth=1)
        plt.title(ct_name.split('/')[-1])
        plt.show()

    if os.path.exists(ct_name):
        print("slice generated")
        return len(plane_select), len(plane_select)

    if len(plane_select) < 4:
        print("slice is insufficient")
        return len(plane_select), len(plane_select)
    else:
        plane_select_start = plane_select[0]
        plane_select_end = plane_select[-1]
        combine_z, combine_z_mask = image_compose_multiple_reverse(np_ct_img, np_mask, x=num_2D[0], z=num_2D[1], to_resize=to_resize, image_resize = image_resize, start_loc=plane_select_start, end_loc=plane_select_end) # for died

        if if_save:
            cv2.imwrite(ct_name, combine_z * 255)
            cv2.imwrite(ct_mask_name, combine_z_mask * 255)

    return combine_z, len(plane_select)



def generate_montage(ct_list, datapath_3D, datapath_3Dmask, savepath_2Dmontage, savepath_2Dmontagemask, num_aug, num_montage, to_resize=False, image_size=[350.350]):

    for ct in ct_list:

        sitkImage = sitk.ReadImage(datapath_3D + ct)
        npImage = sitk.GetArrayFromImage(sitkImage)
        npImage = image_3D_normalisation(npImage)

        # mask
        sitkMask = sitk.ReadImage(datapath_3Dmask + ct)
        npImage_mask = sitk.GetArrayFromImage(sitkMask)

        binary_lung_mask_np = copy.deepcopy(npImage_mask)
        binary_lung_mask_np[binary_lung_mask_np != 0] = 1
        npImage_masked = npImage * binary_lung_mask_np

        '''
        Perform augmentation
        '''
        # index += 1
        for i in range(num_aug):
            print("{} generating".format(ct))
            save_image_path = savepath_2Dmontage + ct.replace('.nii.gz', '_s' + str(i) + '.png')
            save_mask_path = savepath_2Dmontagemask + ct.replace('.nii.gz', '_s' + str(i) + '.png')

            plane_select = [t for t in range(binary_lung_mask_np.shape[0]) if
                            binary_lung_mask_np[t, :, :].sum() > 0]
            if False: # for cambridge lesion mask
                # plot lung and lesion
                lung = [binary_lung_mask_np[t, :, :].sum() for t in plane_select]
                lesion1 = [binary_leision1_mask_np[t, :, :].sum() for t in plane_select]
                lesion2 = [binary_leision2_mask_np[t, :, :].sum() for t in plane_select]

                x_axis = np.linspace(0, 1, len(lung))
                plt.plot(x_axis, lung, color='green', linewidth=1)
                plt.plot(x_axis, lesion1, color='red', linewidth=1)
                plt.plot(x_axis, lesion2, color='blue', linewidth=1)
                plt.title(ct_name.split('/')[-1])
                plt.show()

            # if os.path.exists:
            #     print("slice generated")
            #     continue
                # return len(plane_select), len(plane_select)

            if len(plane_select) < 4:
                print("!!!!!!!!!{}: slice is insufficient".format(ct))
                # return
            else:
                plane_select_start = plane_select[0]
                plane_select_end = plane_select[-1]
                combine_z, combine_z_mask = image_compose_multiple_clockwise(npImage_masked, npImage_mask, x=num_montage[0], z=num_montage[1],
                                                                           to_resize=to_resize,
                                                                           image_resize=image_size,
                                                                           start_loc=plane_select_start,
                                                                           end_loc=plane_select_end)  # for died
                cv2.imwrite(save_image_path, combine_z * 255)
                if False:
                    cv2.imwrite(save_mask_path, combine_z_mask * 255)

    print("Generated Ending")
    return


def generate_montage_two(ct_list, datapath_3D, datapath_3Dmask, savepath_2Dmontage, savepath_2Dmontage_womask, savepath_2Dmontagemask, num_aug,
                     num_montage, to_resize=False, image_size=[350.350]):
    show_index = 0
    # for ct in ct_list[500:]:#y4
    for ct in ct_list:  # y4

        # if ct == '0283.nii.gz':
        #     print("here")
        # else:
        #     continue
        show_index = show_index + 1
        if os.path.exists(savepath_2Dmontage + ct.replace('.nii.gz', '_s49.png')):
            print("Total {}, Making {}".format(len(ct_list), show_index))

            continue
        else:
            print("generating: {}".format(ct))


        try:
            sitkImage = sitk.ReadImage(datapath_3D + ct)
            show_index + 1
        except:
            print("!!!!!!!!!{}: doesn't exist".format(ct))
            show_index + 1
            continue
        npImage = sitk.GetArrayFromImage(sitkImage)
        npImage = image_3D_normalisation(npImage)

        # mask
        sitkMask = sitk.ReadImage(datapath_3Dmask + ct)
        npImage_mask = sitk.GetArrayFromImage(sitkMask)

        binary_lung_mask_np = copy.deepcopy(npImage_mask)
        binary_lung_mask_np[binary_lung_mask_np != 0] = 1
        npImage_masked = npImage * binary_lung_mask_np

        '''
        Perform augmentation
        '''
        # index += 1
        for i in range(num_aug):
            save_image_path = savepath_2Dmontage + ct.replace('.nii.gz', '_s' + str(i) + '.png')
            save_mask_path = savepath_2Dmontagemask + ct.replace('.nii.gz', '_s' + str(i) + '.png')
            save_image_path_womask = savepath_2Dmontage_womask + ct.replace('.nii.gz', '_s' + str(i) + '.png')

            plane_select = [t for t in range(binary_lung_mask_np.shape[0]) if
                            binary_lung_mask_np[t, :, :].sum() > 0]
            if False:  # for cambridge lesion mask
                # plot lung and lesion
                lung = [binary_lung_mask_np[t, :, :].sum() for t in plane_select]
                lesion1 = [binary_leision1_mask_np[t, :, :].sum() for t in plane_select]
                lesion2 = [binary_leision2_mask_np[t, :, :].sum() for t in plane_select]

                x_axis = np.linspace(0, 1, len(lung))
                plt.plot(x_axis, lung, color='green', linewidth=1)
                plt.plot(x_axis, lesion1, color='red', linewidth=1)
                plt.plot(x_axis, lesion2, color='blue', linewidth=1)
                plt.title(ct_name.split('/')[-1])
                plt.show()

            if len(plane_select) < 4:
                print("!!!!!!!!!{}: slice is insufficient".format(ct))
                # return
            else:
                plane_select_start = plane_select[0]
                plane_select_end = plane_select[-1]
                combine_z_womask, combine_z_masked = image_compose_multiple_clockwise_two(npImage,
                                                                                          npImage_masked,
                                                                                          npImage_mask,
                                                                                          x=num_montage[0],
                                                                                          z=num_montage[1],
                                                                                          to_resize=to_resize,
                                                                                          image_resize=image_size,
                                                                                          start_loc=plane_select_start,
                                                                                          end_loc=plane_select_end)
                cv2.imwrite(save_image_path_womask, combine_z_womask * 255)
                cv2.imwrite(save_image_path, combine_z_masked * 255)

    print("Generated Ending")
    return

if __name__ == "__main__":

    setup_seed(20)

    data_ictcf = 1

    num_generate = 50
    num_montage = [2, 2]

    ct_name = 'ID'
    ct_list = ['./patients_enrol_list/enrolled_all.csv']
    datapath_3D = './3D_segmented_lung/'
    datapath_3Dmask = './3D_segmented_mask/'
    savepath_2Dmontage = './2D_montage/' # savepath_2Dmontage_masked
    savepath_2Dmontage_womask = './2D_montage_womask/'
    savepath_2Dmontagemask = savepath_2Dmontage
    cts = set()
    for tb in ct_list:
        data = pd.read_csv(tb)
        cts = cts | set(list(data[ct_name]))
    cts = [str(ct).zfill(4) + '.nii.gz' for ct in cts]

    print(cts)
    generate_montage_two(list(cts), datapath_3D, datapath_3Dmask, savepath_2Dmontage, savepath_2Dmontage_womask, savepath_2Dmontagemask, num_generate, num_montage, to_resize=False, image_size=[350, 350])
