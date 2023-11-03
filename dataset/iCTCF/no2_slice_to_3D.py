import sys
sys.path.append("..")
# from utility import *
from argparse import ArgumentParser
import os
import re
import seaborn as sns
sns.set()
import warnings
warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
# import nibabel as nib
import scipy.ndimage as ndimg
# from myvi import myvi
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt
import math
import datetime
import os, glob, json
import cv2
import copy
import pandas as pd
import pydicom
# import dicom
import scipy
from skimage import measure, morphology
from lungmask import mask
import zipfile
import wget

def resample_image_to_1mm(itk_image, out_spacing=(1.0, 1.0, 1.0), is_label=False):
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()

    out_size = [int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
                int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
                int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))]
    print(out_size)
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)

    return resample.Execute(itk_image)

def resample_image_to_256x256x256(itk_image, is_label=False):
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()

    out_size = [256, 256, 256]
    out_spacing = [original_spacing[0] / (out_size[0] / original_size[0]),
                   original_spacing[1] / (out_size[1] / original_size[1]),
                   original_spacing[2] / (out_size[2] / original_size[2])]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)

    return resample.Execute(itk_image)

def resample_image_to_128x128x128(itk_image, is_label=False):
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()

    out_size = [128, 128, 128]
    out_spacing = [original_spacing[0] / (out_size[0] / original_size[0]),
                   original_spacing[1] / (out_size[1] / original_size[1]),
                   original_spacing[2] / (out_size[2] / original_size[2])]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)

    return resample.Execute(itk_image)

def resample_image_to_128x128x8(itk_image, is_label=False):
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()

    out_size = [128, 128, 8]
    out_spacing = [original_spacing[0] / (out_size[0] / original_size[0]),
                   original_spacing[1] / (out_size[1] / original_size[1]),
                   original_spacing[2] / (out_size[2] / original_size[2])]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)

    return resample.Execute(itk_image)

def resample_image_to_256x256x128(itk_image, is_label=False):
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()

    out_size = [256, 256, 128]
    out_spacing = [original_spacing[0] / (out_size[0] / original_size[0]),
                   original_spacing[1] / (out_size[1] / original_size[1]),
                   original_spacing[2] / (out_size[2] / original_size[2])]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())
    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)

    return resample.Execute(itk_image)

def resample_xy_target_size(itk_image, out_size, is_label=False):
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize() # x, y, z
    out_spacing = [original_spacing[0] / (out_size[2] / original_size[0]), # x
                   original_spacing[1] / (out_size[1] / original_size[1]), # y
                   original_spacing[2] / (original_size[2] / original_size[2])] # z

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize([out_size[1], out_size[2], original_size[2]]) # outsize: z, x, y, setsize: x, y, z
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)

    return resample.Execute(itk_image)

def bbox_3D(img):
    r = np.any(img, axis=(1, 2))
    c = np.any(img, axis=(0, 2))
    z = np.any(img, axis=(0, 1))

    rmin, rmax = np.where(r)[0][[0, -1]]
    cmin, cmax = np.where(c)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]

    return [rmin, rmax, cmin, cmax, zmin, zmax]

def takedate(elem):
    return int(elem)

def bbox_3D(img):
    r = np.any(img, axis=(1, 2))
    c = np.any(img, axis=(0, 2))
    z = np.any(img, axis=(0, 1))

    rmin, rmax = np.where(r)[0][[0, -1]]
    cmin, cmax = np.where(c)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]

    return [rmin, rmax, cmin, cmax, zmin, zmax]

def image_compose(npImage_resample_adjust, x=2, z=2):

    [sz,sx,sy] = npImage_resample_adjust.shape
    combine_x = np.zeros([x * sz, z * sy])
    combine_z = np.zeros([x * sx, z * sy])

    total_num = 0
    for ix in range(x):
        for iz in range(z):
            print(ix,iz)
            combine_x[ix*sz:(ix+1)*sz, iz*sy:(iz+1)*sy] = npImage_resample_adjust[::-1,sx//5*(ix*x+iz+1),::]
            combine_z[ix*sx:(ix+1)*sx, iz*sy:(iz+1)*sy] = npImage_resample_adjust[sz//5*(ix*x+iz+1),:,::]

    return combine_x, combine_z

def get_time_swab(patient_id, clinical_data_feature):

    time1 = clinical_data_feature.loc[clinical_data_feature['Pseudonym'] == patient_id].values[0][
        2+3]  # Date_of_Positive_Covid_Swab
    time2 = clinical_data_feature.loc[clinical_data_feature['Pseudonym'] == patient_id].values[0][
        3+3]  # Date_of_acquisition_of_1st_RT-PCR
    time3 = clinical_data_feature.loc[clinical_data_feature['Pseudonym'] == patient_id].values[0][
        4+3]  # Date_of_acquisition_of_1st_RT-PCR result
    time4 = clinical_data_feature.loc[clinical_data_feature['Pseudonym'] == patient_id].values[0][
        5+3]  # Date_of_acquisition_of_2st_RT-PCR
    time5 = clinical_data_feature.loc[clinical_data_feature['Pseudonym'] == patient_id].values[0][
        6+3]  # Date_of_acquisition_of_2st_RT-PCR


    if not pd.isnull(time1):
        date_slice_str = time1.split(' ')[0]
        yy_slice = int(date_slice_str.split('/')[2])
        mm_slice = int(date_slice_str.split('/')[1])
        dd_slice = int(date_slice_str.split('/')[0])
        data_str = date_slice_str
        date = datetime.datetime(yy_slice, mm_slice, dd_slice)
    elif not pd.isnull(time2):
        date_slice_str = time2.split(' ')[0]
        yy_slice = int(date_slice_str.split('/')[2])
        mm_slice = int(date_slice_str.split('/')[1])
        dd_slice = int(date_slice_str.split('/')[0])
        data_str = date_slice_str
        date = datetime.datetime(yy_slice, mm_slice, dd_slice)
    elif not pd.isnull(time3):
        date_slice_str = time3.split(' ')[0]
        yy_slice = int(date_slice_str.split('/')[2])
        mm_slice = int(date_slice_str.split('/')[1])
        dd_slice = int(date_slice_str.split('/')[0])
        data_str = date_slice_str
        date = datetime.datetime(yy_slice, mm_slice, dd_slice)
    elif not pd.isnull(time4):
        date_slice_str = time4.split(' ')[0]
        yy_slice = int(date_slice_str.split('/')[2])
        mm_slice = int(date_slice_str.split('/')[1])
        dd_slice = int(date_slice_str.split('/')[0])
        data_str = date_slice_str
        date = datetime.datetime(yy_slice, mm_slice, dd_slice)
    elif not pd.isnull(time5):
        date_slice_str = time5.split(' ')[0]
        yy_slice = int(date_slice_str.split('/')[2])
        mm_slice = int(date_slice_str.split('/')[1])
        dd_slice = int(date_slice_str.split('/')[0])
        data_str = date_slice_str
        date = datetime.datetime(yy_slice, mm_slice, dd_slice)
    else:
        # yy_slice = int(2099)
        # mm_slice = int(01)
        # dd_slice = int(01)
        data_str = ''
        date = ''

    return date, data_str

def get_time_ct_scan(scan_name):
    yy_slice = int(scan_name[0:4])
    mm_slice = int(scan_name[4:6])
    dd_slice = int(scan_name[6:8])
    date = datetime.datetime(yy_slice, mm_slice, dd_slice)
    return date

def quality_check(scan_name, data_path, save_image, save_mask, invalidbody=[str("HEART")], total_slice=150, resize_target = [200, 350, 350]):

    j = scan_name
    # data_path_jason = data_path

    # 1. check body
    img_path = data_path + 'Images/' + j
    json_path = data_path + 'Metadata/' + j[0:-7] + '.json'
    lung_path = data_path + 'lung_ggo_consolid/' + j
    save_image = save_image + '/' + j
    save_mask = save_mask + '/' + j

    with open(json_path, 'r') as jsonfile:
        json_dict = json.load(jsonfile)

    if False:
        # sample
        img_stik = sitk.ReadImage(img_path)
        img_np = sitk.GetArrayFromImage(img_stik)
        lung_mask_sitk = sitk.ReadImage(lung_path)
        lung_mask_np = sitk.GetArrayFromImage(lung_mask_sitk)

        # Make the lung mask binary [0, 1, 2, 3] --> [0, 1]
        binary_lung_mask_np = copy.deepcopy(lung_mask_np)
        binary_lung_mask_np[binary_lung_mask_np != 0] = 1

        lung_img_np = img_np
        lung_img_np[binary_lung_mask_np == 0] = -1024


        # Get rid of the backgrounds and leave only lung regions
        try:
            rmin, rmax, cmin, cmax, zmin, zmax = bbox_3D(binary_lung_mask_np)
        except:
            print(j + ": invalidate slice number")
            return 1


        cropped_binary_lung_mask_np = binary_lung_mask_np[rmin:rmax, cmin:cmax, zmin:zmax]
        sum_pixel = cropped_binary_lung_mask_np.shape[1] * cropped_binary_lung_mask_np.shape[2]
        plane_select = [t for t in range(cropped_binary_lung_mask_np.shape[0]) if
                        cropped_binary_lung_mask_np[t, :, :].sum() / sum_pixel > 0.3]
        if len(plane_select) < 20:
            print(j + ": invalidate slice number")
            return 1

        # Crop out Lung
        cropped_lung_region_np = lung_img_np[rmin:rmax, cmin:cmax, zmin:zmax]
        cropped_lung_mask_np   = lung_mask_np[rmin:rmax, cmin:cmax, zmin:zmax]

        # resampled
        cropped_lung_region_sitk = sitk.GetImageFromArray(cropped_lung_region_np)
        cropped_lung_region_sitk.SetSpacing(img_stik.GetSpacing()) # getsize: x, y, z resize_target: z, x, y
        cropped_lung_mask_sitk = sitk.GetImageFromArray(cropped_lung_mask_np)
        cropped_lung_mask_sitk.SetSpacing(lung_mask_sitk.GetSpacing())

        resampled_cropped_lung_region_sitk = resample_xy_target_size(cropped_lung_region_sitk, resize_target)
        resampled_cropped_lung_mask_sitk = resample_xy_target_size(cropped_lung_mask_sitk, resize_target, is_label=True)



        sitk.WriteImage(resampled_cropped_lung_region_sitk, save_image)
        sitk.WriteImage(cropped_lung_mask_sitk, 'check_'+save_mask)
        sitk.WriteImage(resampled_cropped_lung_mask_sitk, save_mask)


    # 2. check kernel
    if str('ConvolutionKernel_1') in json_dict.keys():
        ConvolutionKernel = str(json_dict['ConvolutionKernel_1'])
    elif str('ConvolutionKernel') in json_dict.keys():
        ConvolutionKernel = str(json_dict['ConvolutionKernel'])
    else:
        ConvolutionKernel = "00"
    ck_no = "".join(list(filter(str.isdigit, ConvolutionKernel)))
    if ck_no == 'B':
        return 0
    elif ck_no == '':
        ck_no = "00"

    if not(int(ck_no)>=20 and int(ck_no)<=40):
        print(j + ": invalidate kernel")
        return 2

    return 0

def readtime(time_itu):
    date_slice_str = time_itu.split(' ')[0]
    yy_slice = int(date_slice_str.split('/')[2])
    mm_slice = int(date_slice_str.split('/')[1])
    dd_slice = int(date_slice_str.split('/')[0])
    date = datetime.datetime(yy_slice, mm_slice, dd_slice)
    return date

def load_scan(path):
    slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))

    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness

    return slices

def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    image = image.astype(np.int16)

    image[image == -2000] = 0
    
    for slice_number in range(len(slices)):

        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope

        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)

        image[slice_number] += np.int16(intercept)

    return np.array(image, dtype=np.int16)

def resample(image, scan, new_spacing=[1,1,1]):
    # Determine current pixel spacing
    spacing = np.array([scan[0].SliceThickness]+list(scan[0].PixelSpacing), dtype=np.float32)

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor

    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')

    return image, new_spacing

def plot_3d(image, threshold=-300):

    # Position the scan upright,
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2,1,0)

    verts, faces = measure.marching_cubes_classic(p, threshold)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.70)
    face_color = [0.45, 0.45, 0.75]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()

def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)

    counts = counts[vals != bg]
    vals = vals[vals != bg]

    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None

def segment_lung_mask(image, fill_lung_structures=True):

    # not actually binary, but 1 and 2.
    # 0 is treated as background, which we do not want
    binary_image = np.array(image > -320, dtype=np.int8)+1
    labels = measure.label(binary_image)

    # Pick the pixel in the very corner to determine which label is air.
    #   Improvement: Pick multiple background labels from around the patient
    #   More resistant to "trays" on which the patient lays cutting the air
    #   around the person in half
    background_label = labels[0, 0, 0]

    #Fill the air around the person
    binary_image[background_label == labels] = 2


    # Method of filling the lung structures (that is superior to something like
    # morphological closing)
    if fill_lung_structures:
        # For every slice we determine the largest solid structure
        for i, axial_slice in enumerate(binary_image):
            axial_slice = axial_slice - 1
            labeling = measure.label(axial_slice)
            l_max = largest_label_volume(labeling, bg=0)

            if l_max is not None: #This slice contains some lung
                binary_image[i][labeling != l_max] = 1


    binary_image -= 1 #Make the image actual binary
    binary_image = 1-binary_image # Invert it, lungs are now 1

    # Remove other air pockets insided body
    labels = measure.label(binary_image, background=0)
    l_max = largest_label_volume(labels, bg=0)
    if l_max is not None: # There are air pockets
        binary_image[labels != l_max] = 0

    return binary_image

def all_path(dirname):
    result = []
    for maindir, subdir, file_name_list in os.walk(dirname):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            result.append(apath)
    return result

def load_scan_multiple(path):
    slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path) if s[-4:] == '.dcm']
    slices_serisnum = []
    for slice in slices:
        slices_serisnum.append(slice.SeriesNumber)

    series_select = max(set(slices_serisnum), key=slices_serisnum.count)
    slices_select = [slice for slice in slices if slice.SeriesNumber == series_select]

    slices_select.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices_select[0].ImagePositionPatient[2] - slices_select[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices_select[0].SliceLocation - slices_select[1].SliceLocation)

    for s in slices_select:
        s.SliceThickness = slice_thickness

    return slices_select, len(slices_select)


def load_scan_multiple_same(path):

    # slices = [pydicom.read_file(path + '/' + s,  force=True) for s in os.listdir(path) if s[-4:] == '.dcm']
    slices = []
    for s in os.listdir(path):
        if not s[-4:] == '.dcm':
            continue
        slice = pydicom.read_file(path + '/' + s, force=True)
        try:
            r = slice.InstanceNumber
            slices.append(slice)
            # r = object.__getattribute__(slice, 'ImagePositionPatient')
            # slices.append(slice)
        except:
            print("empty slice")

    slices_serisnum = []
    for slice in slices:
        # print(slice.SeriesNumber, slice.Rows, slice.Columns, slice.InstanceNumber)
        slices_serisnum.append(slice.InstanceNumber)
    loc_max_slices = np.where(slices_serisnum == np.max(slices_serisnum))

    # same series, different slices
    slices_selects = []
    for i in loc_max_slices[0]:

        a = slices[i].SeriesNumber
        b = slices[i].Rows
        c = slices[i].Columns
        print(a,b,c)

        slices_select = [slice for slice in slices if slice.SeriesNumber == a and slice.Rows == b and slice.Columns == c]
        try:
            slices_select.sort(key=lambda x: float(x.ImagePositionPatient[2]))
        except:
            slices_select.sort(key=lambda x: float(x.SliceLocation))

        try:
            slice_thickness = np.abs(slices_select[0].ImagePositionPatient[2] - slices_select[1].ImagePositionPatient[2])
        except:
            slice_thickness = np.abs(slices_select[0].SliceLocation - slices_select[1].SliceLocation)

        for s in slices_select:
            s.SliceThickness = slice_thickness
        slices_selects.append(slices_select)


    return slices_selects

# data
img_folder = "./"

clinical_data = pd.read_csv('./patients_enrol_list/positive_all.csv')
clinical_data = pd.read_csv('./patients_enrol_list/enrolled_all.csv')
patients = clinical_data['ID'].tolist()

count = 0

f =  open('missing.txt', 'w')

deal_num = 0

for patient_name in patients:
    print(deal_num)
    deal_num = deal_num + 1
    if not os.path.isdir(img_folder+"/"+patient_name):
        if os.path.exists(img_folder + "/" + patient_name+'.zip'):
            # count = count + 1
            fz = zipfile.ZipFile(img_folder + "/" + patient_name+'.zip', "r")
            saveroot = './'
            for file in fz.namelist():
                fz.extract(file, saveroot)
            fz.close()
            # url = "https://ngdc.cncb.ac.cn/ictcf/patient/DICOM/Patient%20{}.zip".format(patient_name.split(' ')[-1])
            # f.writelines(url+"\n")

            # print(patient_name + str(os.path.exists(img_folder + "/" + patient_name+'.zip')))
            print("unzipping............."+patient_name)
            patient_path = img_folder + "/" + patient_name + '/CT/'
            # continue
        else:
            count = count + 1
            saveroot = './'
            url = "https://ngdc.cncb.ac.cn/ictcf/patient/DICOM/Patient%20{}.zip".format(patient_name.split(' ')[-1])
            # print(patient_name + " not exist！！！！！！！！！！！！")
            try:
                filename = wget.download(url, out=saveroot)
                print("downloading {}".format(patient_name))
            except:
                # print("download {} failed".format(patient_name))
                delete_index = clinical_data[clinical_data['ID'] == patient_name].index.tolist()
                clinical_data = clinical_data.drop(index=delete_index)
            continue

    else:
        patient_path = img_folder+"/"+patient_name+'/CT/'

    # continue

    for number in range(1):# just consider folder 0, os.listdir(patient_path):

        # print(count)
        imgpath = patient_path
        save_name = patient_name
        save_image_path = './3D_segmented_lung/' + save_name + '.nii.gz'
        save_image_segmentation_path = './3D_segmented_mask/' + save_name + '.nii.gz'

        if os.path.exists(save_image_path):
            continue
            print("here")

        try:
            # patient = load_scan(imgpath)
            patient, num_slices = load_scan_multiple(imgpath)
            patients = [patient]
            # patients = load_scan_multiple_same(imgpath)
        except:
            print('{}: to 3D failture'.format(patient_name))
            continue
        flag_success = 0
        for patient in patients:

            try:
                img_np = get_pixels_hu(patient)
                flag_success = 0
                print('{}: to_pixel success'.format(patient_name))
            except:
                flag_success = 1
                print('{}: to_pixel failture'.format(patient_name))
                delete_index = clinical_data[clinical_data['ID'] == patient_name].index.tolist()
                clinical_data = clinical_data.drop(index=delete_index)
                # continue

            if flag_success == 1:
                continue

            # lung mask
            lung_mask_np = mask.apply(img_np)
            binary_lung_mask_np = copy.deepcopy(lung_mask_np)
            binary_lung_mask_np[binary_lung_mask_np != 0] = 1

            # cropped lung
            try:
                rmin, rmax, cmin, cmax, zmin, zmax = bbox_3D(binary_lung_mask_np)
            except:
                print("{}: crop failure".format(patient_name))
                continue

            cropped_img_np = img_np[rmin:rmax, cmin:cmax, zmin:zmax]
            cropped_lung_mask_np = lung_mask_np[rmin:rmax, cmin:cmax, zmin:zmax]
            cropped_binary_lung_mask_np = binary_lung_mask_np[rmin:rmax, cmin:cmax, zmin:zmax]

            # resample
            z_size = cropped_lung_mask_np.shape[0]
            cropped_img_np_shrink = np.zeros([z_size, 350, 350])
            cropped_lung_mask_np_shrink = np.zeros([z_size, 350, 350])
            for z in range(z_size):
                cropped_img_np_shrink[z, :, :] = cv2.resize(cropped_img_np[z,:,:], [350, 350], interpolation=cv2.INTER_AREA)
                cropped_lung_mask_np_shrink[z, :, :] = cv2.resize(cropped_lung_mask_np[z,:,:], [350, 350], interpolation=cv2.INTER_NEAREST)


            cropped_img_np_shrink_sitk = sitk.GetImageFromArray(cropped_img_np_shrink)
            cropped_lung_mask_np_shrink_sitk = sitk.GetImageFromArray(cropped_lung_mask_np_shrink)


            sitk.WriteImage(cropped_img_np_shrink_sitk, save_image_path)
            sitk.WriteImage(cropped_lung_mask_np_shrink_sitk, save_image_segmentation_path)


print("we are here")

print(count)
f.close()

# clinical_data.to_csv('./patients_enrol_list/positive_all_both.csv')
patient_info_enrolled_died = clinical_data[clinical_data['Mortality outcome']=='Deceased']
patient_info_enrolled_survived = clinical_data[clinical_data['Mortality outcome']=='Cured']

patient_info_enrolled_mild = clinical_data[clinical_data['Morbidity outcome']=='Mild']
patient_info_enrolled_regular = clinical_data[clinical_data['Morbidity outcome']=='Regular']
patient_info_enrolled_severe = clinical_data[clinical_data['Morbidity outcome']=='Severe']
patient_info_enrolled_criticallyill = clinical_data[clinical_data['Morbidity outcome']=='Critically ill']

patient_info_enrolled_0 =   pd.concat([patient_info_enrolled_mild,patient_info_enrolled_regular])
patient_info_enrolled_1 =   pd.concat([patient_info_enrolled_severe,patient_info_enrolled_criticallyill])
patient_info_enrolled_01  = pd.concat([patient_info_enrolled_0,patient_info_enrolled_1])
patient_info_enrolled_0.to_csv('./patients_enrol_list/enrolled_0.csv')
patient_info_enrolled_1.to_csv('./patients_enrol_list/enrolled_1.csv')
patient_info_enrolled_01.to_csv('./patients_enrol_list/enrolled_all.csv')
print("0: {}, 1: {}".format(len(patient_info_enrolled_0), len(patient_info_enrolled_1)))

# patient_info_enrolled_died.to_csv('./patients_enrol_list/enrolled_died_both.csv')
# patient_info_enrolled_survived.to_csv('./patients_enrol_list/enrolled_survived_both.csv')
# patient_info_enrolled_mild.to_csv('./patients_enrol_list/enrolled_mild_both.csv')
# patient_info_enrolled_regular.to_csv('./patients_enrol_list/enrolled_regular_both.csv')
# patient_info_enrolled_severe.to_csv('./patients_enrol_list/enrolled_severe_both.csv')
# patient_info_enrolled_criticallyill.to_csv('./patients_enrol_list/enrolled_criticallyill_both.csv')