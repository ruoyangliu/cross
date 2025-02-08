import correct_covert_gt_from_minc_to_itk as correct
import SimpleITK as sitk
from sklearn.model_selection import train_test_split
from itertools import product
import os
import numpy as np
import shutil   
import re
import os.path as path
import random
import sys
import torch
sys.path.append('/home/rl23/Desktop/code/dummy/Attention-Reg')
from utils import data_loading_funcs as load_func
def get_us_gt():
    image_path='/home/rl23/Desktop/data/RESECT/RESECT/NIFTI'
    specific_image_name='US-before.nii.gz'
    us_img_dict=correct.load_img(image_path,specific_image_name)
    print('us_img_dict',us_img_dict.keys())
    #get the gt us image from the us image by uisng lps trnasformation
    for img in us_img_dict:
        img_numer=img.split('-')[0].split('e')[-1]
        if img_numer=='11':
            continue
        print('img_numer',img_numer)
        image=us_img_dict[img]
        print(f'/home/rl23/Desktop/data/CorrectTransformationPipeline/image/Case{img_numer}/sitk_lps/{img_numer}_lpsItk.tfm')
        gt=sitk.ReadTransform(f'/home/rl23/Desktop/data/CorrectTransformationPipeline/image/Case{img_numer}/sitk_lps/{img_numer}_lpsItk.tfm')

        gt_us=sitk.Resample(image,gt)
        correct.save_img_to_file(gt_us,f'/home/rl23/Desktop/data/CorrectTransformationPipeline/image/Case{img_numer}/lpsgt_us/{img_numer}_us_lpsItk.nii.gz')


def split_data(data_dict):
    
    img_names= list(data_dict.keys())
    #print('img_names',img_names)
   
    train_name, test_name = train_test_split(img_names, test_size=0.1,random_state=42)
    #print('train_name',train_name)
    train_name, val_name=train_test_split(train_name, test_size=0.1/0.9,random_state=42)

    train_set={name: data_dict[name] for name in train_name}
    test_set={name: data_dict[name] for name in test_name}
    val_set={name: data_dict[name] for name in val_name}

    return train_set, test_set, val_set




def get_bounding_box(mask):
    # Find indices where mask is 1
    non_zero_indices = np.argwhere(mask)
    # Get the minimum and maximum indices along each dimension
    min_idx = non_zero_indices.min(axis=0)
    max_idx = non_zero_indices.max(axis=0)
    return min_idx, max_idx

def get_physical_coordinates(image_us,image_mri, corners):
    mri_points=[]
    us_point=[]
    for corner in corners:
        us_points=image_us.TransformIndexToPhysicalPoint(corner)
        us_point.append(us_points)
        mri_points.append(image_mri.TransformPhysicalPointToIndex(us_points))
    
    return mri_points

# input orginal mri and us and return croped mri
def crop_mri(img_mri,img_us):

    correct_size_mri=[]
    
    us_array=sitk.GetArrayFromImage(img_us)
    binary_mask = (us_array!= 0).astype(np.int16)
    #print('binary_mask',binary_mask.shape)
    min_coordinates, max_coordinates = get_bounding_box(binary_mask)
    min_coordinates=min_coordinates.tolist()
    max_coordinates=max_coordinates.tolist()

   # print("Bounding Box Coordinates:")
    #print("Min Coordinates: ", min_coordinates)
    #print("Max Coordinates: ", max_coordinates)
    corners=list(product(*zip(min_coordinates, max_coordinates)))
    #print('corners',corners)
    mri_points=get_physical_coordinates(img_us,img_mri,corners)
    #print('mri_points',mri_points)
    mri_points_np = np.array(mri_points)
    correct_nmin = mri_points_np.min(axis=0)
    correct_nmin=np.clip(correct_nmin,0,img_mri.GetSize())
    correct_nmin=correct_nmin.tolist()
    
    correct_nmax = mri_points_np.max(axis=0)
    correct_nmax=np.clip(correct_nmax,0,img_mri.GetSize())
    correct_nmax=correct_nmax.tolist()
    
    #print("Bounding Box Coordinates in MRI Space:")
    #print("Min Coordinates: ", correct_nmin)
    #print("Max Coordinates: ", correct_nmax)

    for i in range(len(correct_nmax)):
        correct_size_mri.append(correct_nmax[i] - correct_nmin[i])
    #print("Size Coordinates: ", correct_size_mri)
    #print(type(correct_size_mri))

    
    crop=sitk.CropImageFilter()
    crop.SetLowerBoundaryCropSize(correct_nmin)
    og = list(img_mri.GetSize()) 
    # Assuming correct_nmax is also a list, we subtract corresponding elements
    max_boundary_crop_size = [og[i] - correct_nmax[i] for i in range(len(correct_nmax))]
    #case13
    #print(max_boundary_crop_size)
    crop.SetUpperBoundaryCropSize(max_boundary_crop_size)
    cropped_mri_sitk=crop.Execute(img_mri)
    print('cropped_mri_sitk',cropped_mri_sitk.GetSize())

    
    return cropped_mri_sitk


def save_patch_np_us(patches,patch_name,output_path):
        print(patches)
        os.makedirs(output_path,exist_ok=True)

        #print(output_path)
        #sitk.WriteImage(patches[i],os.path.join(output_path,f'resample_us_{i}.nii.gz'))
        np.save(os.path.join(output_path,f'{patch_name}'),patches)

def save_patch_np_mri(patches,patch_name,output_path):
        print(patches)
        os.makedirs(output_path,exist_ok=True)
            #print(output_path)
            #sitk.WriteImage(patches[i],os.path.join(output_path,f'resample_mri_{i}.nii.gz'))
        np.save(os.path.join(output_path,f'{patch_name}'),patches)

def save_patch_nii_us(patches,patch_name,output_path):

        os.makedirs(output_path,exist_ok=True)
        
        #print(output_path)
        sitk.WriteImage(patches,os.path.join(output_path,f'{patch_name}'))
        #np.save(os.path.join(output_path,f'resample_us_{i}'),patches[i])

def save_patch_nii_mri(patches,patch_name,output_path):

        os.makedirs(output_path,exist_ok=True)
        
        #print(output_path)
        sitk.WriteImage(patches,os.path.join(output_path,f'{patch_name}'))
        #np.save(os.path.join(output_path,f'resample_us_{i}'),patches[i])

def resize_image_to_fixed_size(image, target_size=(96, 96, 96)):
    """
    Resize the input image to the target size while keeping the same Field of View (FOV).
    
    Parameters:
        image (sitk.Image): Input SimpleITK image.
        target_size (tuple): Desired size (width, height, depth) in voxels.
        
    Returns:
        sitk.Image: Resized image with the same FOV.
    """
    # Get original size and spacing
    original_size = image.GetSize()  # (width, height, depth)
    original_spacing = image.GetSpacing()  # (spacing_x, spacing_y, spacing_z)

    # Compute the new spacing to maintain the same FOV
    target_spacing = [
        (original_size[i] * original_spacing[i]) / target_size[i]
        for i in range(3)
    ]
    
    # Define the resampling filter
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(tuple(target_spacing))
    resampler.SetSize(target_size)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetInterpolator(sitk.sitkLinear)  # Use linear interpolation for resizing
    resampler.SetDefaultPixelValue(image.GetPixelIDValue())  # Background value if necessary
    
    # Apply the resampling
    resized_image = resampler.Execute(image)
    
    return resized_image

     

def excute_cropmri_96():
    image_path='/home/rl23/Desktop/data/RESECT/RESECT/NIFTI'
    specific_image_name='T1.nii.gz'
    mri_img=correct.load_img(image_path,specific_image_name)
    print('mri_img',mri_img.keys())
    image_path='/home/rl23/Desktop/data/RESECT/RESECT/NIFTI'
    specific_image_name='US-before.nii.gz'
    us_img=correct.load_img(image_path,specific_image_name)
    croped_mri_96_list={}
    for name in mri_img.keys():
        print(name)
        if name=='11':
            continue
        #print('case',mri_img[name])
        croped_mri=crop_mri(mri_img[name],us_img[name])
        #save_patch_nii_mri(croped_mri,f'Case{name}_cropped_mri.nii.gz',f'/home/rl23/Desktop/data/CorrectTransformationPipeline/Case{name}/croped_mri/')
        print('croped_mri',croped_mri.GetSize())
        
        croped_mri_96=resize_image_to_fixed_size(croped_mri)
        save_patch_nii_mri(croped_mri_96,f'Case{name}_croped_mri_96.nii.gz',f'/home/rl23/Desktop/data/CorrectTransformationPipeline/Case{name}/croped_mri_96/')
        croped_mri_96_list[name]=croped_mri_96
    #save_patch_nii_mri(croped_mri_96_list['7'],f'Case7_cropped_mri_96.nii.gz',f'/home/rl23/Desktop/data/CorrectTransformationPipeline/Case7/croped_mri_96/')

def excute_gt_us_patch():
    image_path='/home/rl23/Desktop/data/RESECT/RESECT/NIFTI'
    specific_image_name='US-before.nii.gz'
    us_img=correct.load_img(image_path,specific_image_name)

    image_path='/home/rl23/Desktop/data/CorrectTransformationPipeline/image'
    specific_image_name='croped_mri_96.nii.gz'
    mri_img=correct.load_img(image_path,specific_image_name)
    print('mri_img',mri_img.keys())
    path_us={}
    for name in mri_img.keys():
        print(name)
        print('mri',mri_img[name].GetSize())
        gt_path=f'/home/rl23/Desktop/data/CorrectTransformationPipeline/Case{name}/sitk_lps/{name}_lpsItk.tfm'
        print('gt_path',gt_path)
        gt=sitk.ReadTransform(gt_path)
        gt_us=sitk.Resample(us_img[name],mri_img[name],gt)
        print('mri',mri_img[name].GetSize())
        print('gt_us',gt_us.GetSize())
        print('mri',mri_img[name].GetSpacing())
        print('gt_us',gt_us.GetSpacing())
        print('mri',mri_img[name].GetOrigin())
        print('gt_us',gt_us.GetOrigin())
        path_us[name]=gt_us
    print('path_us',path_us.keys())
    sitk.WriteImage(path_us['2'],f'/home/rl23/Desktop/data/CorrectTransformationPipeline/Case2/gt_us_96/2_gt_us_96.nii.gz')

def excute_gt_us_resample():
    img_path='/home/rl23/Desktop/data/CorrectTransformationPipeline'
    us_specific_name='us_lpsItk.nii.gz'
    us_patch=correct.load_img(img_path,us_specific_name)
    mri_specific_name='croped_mri_96.nii.gz'
    mri_patch=correct.load_img(img_path,mri_specific_name)
    print('us_patch',us_patch.keys())
    print('mri_patch',mri_patch.keys())
    
    us_resample_list={}
    for name in us_patch.keys():
         
        resampler = sitk.ResampleImageFilter()
        resampler.SetSize(mri_patch[name].GetSize())
        resampler.SetOutputSpacing(mri_patch[name].GetSpacing())
        resampler.SetOutputDirection(mri_patch[name].GetDirection())
        resampler.SetOutputOrigin(mri_patch[name].GetOrigin())
        us_re=resampler.Execute(us_patch[name])
        us_resample_list[name]=us_re
        save_patch_nii_mri(us_re,f'Case{name}_gtus_resample.nii.gz',f'/home/rl23/Desktop/data/CorrectTransformationPipeline/Case{name}/gtus_resample_mri96/')
    #sitk.WriteImage(us_resample_list['8'], f'/home/rl23/Desktop/data/CorrectTransformationPipeline/Case8/us_resample_mri96/Case8_us_resample.nii.gz')

def apply_new_trans_us():
     
    us=sitk.ReadImage('/home/rl23/Desktop/data/CorrectTransformationPipeline/image/Case2/gtus_resample_mri96/Case2_gtus_resample.nii.gz')
    mri=sitk.ReadImage('/home/rl23/Desktop/data/CorrectTransformationPipeline/image/Case2/croped_mri_96/Case2_croped_mri_96.nii.gz')
    #find mask of us
    us_array=sitk.GetArrayFromImage(us)
    binary_mask = (us_array!= 0).astype(np.int16)
    us_mask=sitk.GetImageFromArray(binary_mask)

    us_mask.SetDirection(us.GetDirection())
    us_mask.SetOrigin(us.GetOrigin())
    us_mask.SetSpacing(us.GetSpacing())
    #sitk.WriteImage(us_mask,'/home/rl23/Desktop/data/CorrectTransformationPipeline/Case2/gtus_resample_mri96/Case2_gtus_resample_mask.nii.gz')

    # get the us part center in LPS
    
    label_stats = sitk.LabelShapeStatisticsImageFilter()
    label_stats.Execute(us_mask)
    centroid = label_stats.GetCentroid(1)
    print(centroid)
    # set new transformation
    new_tran=sitk.Euler3DTransform()
    new_tran.SetCenter(centroid)
    new_tran.SetTranslation([0,0,0])
    new_tran.SetRotation(0,np.radians(45),0)

    matrix=new_tran.GetMatrix()
    translation=new_tran.GetTranslation()

    affine=sitk.AffineTransform(3)
    affine.SetMatrix(matrix)
    affine.SetTranslation(translation)
    affine.SetCenter(centroid)

    sitk.WriteTransform(affine,'/home/rl23/Desktop/data/CorrectTransformationPipeline/image/Case2/sitk_lps/Case2_gtus_resample_rotation_y+45_affine.tfm')

    us_re=sitk.Resample(us, affine,sitk.sitkLinear,0.0,us.GetPixelID())
    sitk.WriteImage(us_re,'/home/rl23/Desktop/data/CorrectTransformationPipeline/image/Case2/gtus_resample_mri96/Case2_gtus_resample_rotation_y+45_affine.nii.gz')




def landmark_copy():
    landmarks=[]
    file_path='/home/rl23/Desktop/data/CorrectTransformationPipeline/landmark_RAS'
    file='/home/rl23/Desktop/data/CorrectTransformationPipeline/test_data'
    final_np_path=os.path.join(file,'train') 
    #print(final_np_path)
    landmarks=os.path.join(file_path,'landmarks')
    landmarks=sorted([f for f in os.listdir(file_path) if f.endswith('.fcsv') ], key=lambda x: int(re.search(r'\d+', x).group()))
    print(landmarks)
    for landmark in landmarks:
        case_id=landmark.split('-')[0]
        
        for file_name in os.listdir(final_np_path):
            folder_path = os.path.join(final_np_path, file_name)
            upp=file_name[0].upper()
            final=upp+file_name[1:]
            print(final)
            if final == case_id:
                #print(file_name)
                source_file_path = os.path.join(file_path, landmark)
                destination_file_path = os.path.join(folder_path, landmark)
                shutil.copy(source_file_path, destination_file_path)
    
def resmaple_us_mri96():
    image_path='/home/rl23/Desktop/data/RESECT/RESECT/NIFTI'
    specific_image_name='US-before.nii.gz'
    us_img=correct.load_img(image_path,specific_image_name)

    image_path='/home/rl23/Desktop/data/CorrectTransformationPipeline'
    specific_image_name='croped_mri_96.nii.gz'
    mri_img=correct.load_img(image_path,specific_image_name)
    print('mri_img',mri_img.keys())
    path_us={}
    for name in mri_img.keys():
        #gt_path=f'/home/rl23/Desktop/data/CorrectTransformationPipeline/Case{name}/sitk_lps/{name}_lpsItk.tfm'
        #print('gt_path',gt_path)
        #gt=sitk.ReadTransform(gt_path)
        print('name',name)
        re_us=sitk.Resample(us_img[name],mri_img[name])
        print('mri',mri_img[name].GetSize())
        print('gt_us',re_us.GetSize())
        print('mri',mri_img[name].GetSpacing())
        print('gt_us',re_us.GetSpacing())
        print('mri',mri_img[name].GetOrigin())
        print('gt_us',re_us.GetOrigin())
        #path_us[name]=gt_us
        print('path_us',path_us.keys())
        save_patch_nii_mri(re_us,f'{name}_gt_us_96.nii.gz',f'/home/rl23/Desktop/data/CorrectTransformationPipeline/image/Case{name}/us_96/')






def _get_random_value(r, center, hasSign):

    
    
    randNumber = random.random() * r + center


    if hasSign:
        sign = random.random() > 0.5
        if sign == False:
            randNumber *= -1

    return randNumber

def generate_random_transform():
 
    
    # Get random rotation and translation
    # The hard coded values are based on the statistical analysis of
    # euler_angle = 13 * np.pi / 180

    signed = True

    euler_angle = 10
    #angleX = 45
    angleX = _get_random_value(euler_angle, 0, signed)
    angleY = _get_random_value(euler_angle, 0, signed)
    angleZ = _get_random_value(euler_angle, 0, signed)

    translation_range =5
    tX = _get_random_value(translation_range, 0, signed)
    tY = _get_random_value(translation_range, 0, signed)
    tZ = _get_random_value(translation_range, 0, signed)

    

    parameters = np.asarray([angleX, angleY, angleZ, tX, tY, tZ])
    #print('parameters',parameters)
    #arrTrans = load_func.construct_matrix_degree(parameters,
    #                                            initial_transform=base_trans_mat4x4)

    # afTrans now is affine transformation matrix
    arrTrans=load_func.construct_matrix(parameters)
    #print('arrTrans',arrTrans)

    return arrTrans, parameters

# Load data


def generate_random_trans():
    image_path='/home/rl23/Desktop/data/CorrectTransformationPipeline/test_data/val'
    specific_image_name='croped_mri_96.nii.gz'
    mri_img=correct.load_img(image_path,specific_image_name)

    specific_image_name_us='gtus_resample.nii.gz'
    us_gt=correct.load_img(image_path,specific_image_name_us)
    print('mri_img',mri_img.keys())
    print('us_gt',us_gt.keys())
    for name in mri_img.keys():
        print('name',name)
        gt_path=f'/home/rl23/Desktop/data/CorrectTransformationPipeline/image/Case{name}/sitk_lps/{name}_lpsItk.tfm'
        #print('gt_path',gt_path)
        gt,gt_param=load_func.gt_load(gt_path)
        
        new_gt=generate_random_transform()
        #print('new_gt',new_gt)
        us_array=sitk.GetArrayFromImage(us_gt[name])
        binary_mask = (us_array!= 0).astype(np.int16)
        us_mask=sitk.GetImageFromArray(binary_mask)

        us_mask.SetDirection(us_gt[name].GetDirection())
        us_mask.SetOrigin(us_gt[name].GetOrigin())
        us_mask.SetSpacing(us_gt[name].GetSpacing())
        label_stats = sitk.LabelShapeStatisticsImageFilter()
        label_stats.Execute(us_mask)
        centroid = label_stats.GetCentroid(1)
        print(centroid)
        
        new_gt[0].SetCenter(centroid)
        print('new_gt',new_gt[0])


        comb=sitk.CompositeTransform([gt,new_gt[0]])
        sitk.WriteTransform(comb,f'/home/rl23/Desktop/data/CorrectTransformationPipeline/test_data/val/Case{name}/{name}_lpsItk_initial_r10_t5.tfm')

def copy_us_orginal():
    image_path='/home/rl23/Desktop/data/RESECT/RESECT/NIFTI'
    specific_image_name='US-before.nii.gz'
    us_img=correct.load_img(image_path,specific_image_name)

    image_path='/home/rl23/Desktop/data/CorrectTransformationPipeline/test_data/train'
    specific_image_name='croped_mri_96.nii.gz'
    mri_img=correct.load_img(image_path,specific_image_name)
    print('mri_img',mri_img.keys())

    for name in mri_img.keys():
        
        print('us',us_img[name].GetSize())
        sitk.WriteImage(us_img[name],f'/home/rl23/Desktop/data/CorrectTransformationPipeline/test_data/train/Case{name}/{name}_us_orginal.nii.gz')


if __name__ == '__main__':
    #excute_cropmri_96()
    #excute_gt_us_patch()
    #excute_gt_us_resample()
    #apply_new_trans_us()
    #landmark_copy()
    #resmaple_us_mri96()
    generate_random_trans()
    #copy_us_orginal()








