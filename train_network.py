import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset
import numpy as np
import time
import os
from os import path
import random
from utils import data_loading_funcs as load_func
import SimpleITK as sitk
from networks import generators as gens
#from networks import small_gen as gens
from numpy.linalg import inv
from datetime import datetime
import argparse
from sklearn.model_selection import train_test_split
import math
from torch.optim.lr_scheduler import CyclicLR
################
print(torch.__version__)
desc = 'Training registration generator'
parser = argparse.ArgumentParser(description=desc)
parser.add_argument('-i', '--init_mode',
                    type=str,
                    help="mode of training with different transformation matrics",
                    default='load')

parser.add_argument('-l', '--learning_rate',
                    type=float,
                    help='Learning rate',
                    default=0.0001) #we used 0.001

parser.add_argument('-d', '--device_no',
                    type=int,
                    choices=[0, 1, 2, 3, 4, 5, 6, 7],
                    help='GPU device number [0-7]',
                    default=0)

parser.add_argument('-e', '--epochs',
                    type=int,
                    help='number of training epochs',
                    default=10) #we used 300 on our dataset

parser.add_argument('-n', '--network_type',
                    type=str,
                    help='choose different network architectures',
                    default='AttentionReg')

parser.add_argument('-info', '--infomation',
                    type=str,
                    help='infomation of this round of experiment',
                    default='None')
parser.add_argument('-b', '--batch_size',
                    type=int,
                    help='batch size',
                    default=1
                    
                    )

net = 'Generator'

#we used 8 or 16 in our experiments
# print('batch size = ',batch_size)
current_epoch = 0
args = parser.parse_args()
batch_size = args.batch_size 
device_no = args.device_no
epochs = args.epochs
device = torch.device("cuda:{}".format(device_no))


def filename_list(dir):
    images = []
    dir = os.path.expanduser(dir)
    # print('dir {}'.format(dir))
    for filename in os.listdir(dir):
        # print(filename)
        file_path = path.join(dir, filename)
        images.append(file_path)
        # print(file_path)
    # print(images)
    return images

import numpy as np

def scale_volume_std(input_volume, upper_bound=255, lower_bound=0):
    mean_value = np.mean(input_volume)
    std_dev = np.std(input_volume)
    
    if std_dev == 0:
        # To avoid division by zero if the volume is constant
        return np.full_like(input_volume, lower_bound)
    
    # Standard deviation normalization (Z-score)
    z_score_volume = (input_volume - mean_value) / std_dev
    
    # Optional: Rescale to the desired range [lower_bound, upper_bound]
    min_z = np.min(z_score_volume)
    max_z = np.max(z_score_volume)
    
    k = (upper_bound - lower_bound) / (max_z - min_z)
    scaled_volume = k * (z_score_volume - min_z) + lower_bound
    
    # Uncomment to debug the range
    # print('min of scaled:', np.min(scaled_volume))
    # print('max of scaled:', np.max(scaled_volume))
    
    return scaled_volume




def scale_volume(input_volume, upper_bound=255, lower_bound=0):
    max_value = np.max(input_volume)
    min_value = np.min(input_volume)

    k = (upper_bound - lower_bound) / (max_value - min_value)  #here try standard deviation instead
    scaled_volume = k * (input_volume - min_value) + lower_bound
    # print('min of scaled {}'.format(np.min(scaled_volume)))
    # print('max of scaled {}'.format(np.max(scaled_volume)))
    return scaled_volume

class Evaluator:
    '''
    the gt should be load as lps and landmarks should be converted to lps before calculating the TRE
    '''
    def __init__(self, reference_landmarks_mri, reference_landmarks_us):
        """
        this function run once at begining 
        Evaluator to compute the TRE (Target Registration Error).
        :param reference_landmarks: Array of ground-truth landmarks, shape (N, 3), where N is the number of landmarks.
        """
        self.reference_landmarks_mri = np.copy(reference_landmarks_mri)
        self.reference_landmarks_us = np.copy(reference_landmarks_us)

    def apply_transformation(self, landmarks, transform_matrix):
        """
        Apply a transformation matrix to a set of landmarks.
        :param landmarks: Array of landmarks (N, 3) to be transformed.
        :param transform_matrix: 4x4 transformation matrix to apply to the landmarks.
        :return: Transformed landmarks, shape (N, 3).
        """
        
        #convert the landmarks to lps
        lps_landmarks=np.copy(landmarks)
        #print('landmarks',type(landmarks))
        #convert the landmarks to lps
        for points in lps_landmarks:
            points[0]=points[0]*-1
            points[1]=points[1]*-1
            points[2]=points[2]


        #tansformed the landmarks = inverse of transform from image 
        transformed_landmarks=[transform_matrix.GetInverse().TransformPoint(points) for points in lps_landmarks]
        
        #print('transformed_landmarks',transformed_landmarks)
        return transformed_landmarks

    def compute_TRE(self,landmarks, transformed_landmarks):
        """
        Compute the Target Registration Error (TRE).
        :param transformed_landmarks: Transformed landmarks, shape (N, 3).
        :return: TRE value (Euclidean distance).
        """
        # Compute the Euclidean distance between transformed landmarks and ground truth landmarks
        #print('landmark_mri',landmark)
        lps_landmarks=np.copy(landmarks)
        for points in lps_landmarks:
            points[0]=points[0]*-1
            points[1]=points[1]*-1
            points[2]=points[2]
        minc_distances = [np.linalg.norm(np.array(t) - np.array(m)) for t, m in zip(lps_landmarks, transformed_landmarks)]
        
        tre = np.mean(minc_distances)
        
        return tre

    def evaluate_transform(self, transform_matrix):
        """
        Evaluate the transformation matrix by applying it to the reference landmarks and calculating TRE.
        :param transform_matrix: 4x4 transformation matrix.
        :return: TRE value (average distance between transformed and reference landmarks).
        """
        #us2mri
        transformed_landmarks_us = self.apply_transformation(self.reference_landmarks_us, transform_matrix)
        #print('transformed_landmarks_mri',transformed_landmarks_mri)
        return self.compute_TRE(self.reference_landmarks_mri,transformed_landmarks_us),transformed_landmarks_us





class MR_TRUS_4D(Dataset):

    def __init__(self, root_dir, initialization):
        """
        in here load all data and split the train , test and val 
        """
        samples = filename_list(root_dir)

        """list with all samples"""
        if root_dir[-3:]  == 'val':
            self.status= 'val'
        else:
            self.status= 'train'
        self.samples = samples
        self.initialization = initialization

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        #print('idx {}'.format(idx))
        """
        :param idx:
        :return:
        """

        case_folder = self.samples[idx]
        #print('case_folder {}'.format(case_folder))
        case_id = case_folder.split('e')[-1]
        #print('case_id {}'.format(case_id))
        index = int(case_id)
        #print('index {}'.format(index))
        

        norm_path = path.normpath(case_folder)
        res = norm_path.split(os.sep)
        status = res[-2]


        """ Load landmarks """

        mri_landmarks_file = os.path.join(case_folder, f'Case{case_id}-MRI.fcsv')
        us_landmarks_file = os.path.join(case_folder, f'Case{case_id}-before_US.fcsv')
        mri_landmark=load_func.load_landmarks(mri_landmarks_file)
        us_landmark=load_func.load_landmarks(us_landmarks_file)

        ''' Load ground-truth registration '''

        # apply to image is using lps coordinate system and landmrks are ras and use ras gt to calculate the TRE
        gt_trans_fn = os.path.join(case_folder, f'{case_id}_lpsItk.tfm')  
        gt, gt_params=load_func.gt_load(gt_trans_fn)
        
        #print('gt_mat {}'.format(gt_mat))
        #print('gt_params {}'.format(gt_params))

        """generated random purtabation"""
        if self.initialization == 'load':
            # Although the training set is generated afresh, we recommend using the
            # same validation set from epoch to epoch for stability. However, we cannot upload that much
            # files, so we will use random validation samples in this demo.
            if status == 'val':
                #new_initial_transform = sitk.ReadTransform(os.path.join(case_folder, f'{case_id}_lpsItk_initial_r5_t5.tfm'))

                # the parameters are in degrees and mm
                tran_path=os.path.join(case_folder, f'{case_id}_lpsItk_initial_r5_t5.tfm')
                #print('tran_path {}'.format(tran_path))
                #print('tran_path {}'.format(tran_path))
                new_initial_transform = sitk.ReadTransform(tran_path)
                params_rand=np.asarray(new_initial_transform.GetNthTransform(1).GetParameters())
                angle_x=math.degrees(params_rand[0])
                angle_y=math.degrees(params_rand[1])
                angle_z=math.degrees(params_rand[2])
                params_rand=np.asarray([angle_x,angle_y,angle_z,params_rand[3],params_rand[4],params_rand[5]])
                us_gt=sitk.ReadImage(path.join(case_folder, f'Case{case_id}_gtus_resample.nii.gz'))
                us_center=load_func.find_us_center(us_gt)
                #print(params_rand)
            # base_mat, params_rand = generate_random_transform()

            # us_gt=sitk.ReadImage(path.join(case_folder, f'Case{case_id}_gtus_resample.nii.gz'))
            # us_center=load_func.find_us_center(us_gt)
            # base_mat.SetCenter(us_center)
            # new_initial_transform = sitk.CompositeTransform([gt,base_mat])

        elif self.initialization == 'random_uniform':
            #generate samples with random SRE in a certain range (e.g. [0-20] or [0-8])
            # if you are provided with ground truth segmentation, calculate
            # the randomized base_TRE (Target Registration Error):
            # base_TRE = evaluator.evaluate_transform(base_mat)

            evaluator=Evaluator(mri_landmark, us_landmark)
            gt_tre=evaluator.evaluate_transform(gt)
            #print('gt_tre {}'.format(gt_tre))
            base_mat, params_rand = generate_random_transform()
            base_TRE = evaluator.evaluate_transform(base_mat)
            uniform_target_TRE = np.random.uniform(0, 20, 1)[0]
            scale_ratio = uniform_target_TRE / base_TRE
            params_rand = params_rand * scale_ratio
            base_mat = load_func.construct_matrix_degree(params=params_rand
                                                    )

        else:
            print('!' * 10 + ' Initialization mode <{}> not supported!'.format(self.initialization))
            return

        """loading MR and US images. In our experiments, we read images from mhd files and resample them with MR segmentation."""
        sample4D = np.zeros((2, 96, 96, 96), dtype=np.ubyte)
        #print('case_folder {}'.format(case_folder))
        #np1=np.load(path.join(case_folder, 'mri.npy'))
        #mri and us are center overlapped 
        
        mri=sitk.ReadImage(path.join(case_folder, f'Case{case_id}_croped_mri_96.nii.gz'))
        us=sitk.ReadImage(path.join(case_folder, f'{case_id}_us_orginal.nii.gz'))
        if self.status == 'train':
            # To randomly generate the transformation matrices
            base_mat, params_rand = generate_random_transform()

            us_gt=sitk.ReadImage(path.join(case_folder, f'Case{case_id}_gtus_resample.nii.gz'))
            us_center=load_func.find_us_center(us_gt)
            base_mat.SetCenter(us_center)
            new_initial_transform = sitk.CompositeTransform([gt,base_mat])
            

        #sitk.WriteImage(us,path.join('/home/rl23/Desktop/data/CorrectTransformationPipeline/image_check', f'us_gt{case_id}.nii.gz'))
        us=sitk.Resample(us,mri,new_initial_transform, sitk.sitkLinear,0.0,mri.GetPixelID())
        #sitk.WriteImage(us,path.join('/home/rl23/Desktop/data/CorrectTransformationPipeline/image_check', f'us_gt_resampled{case_id}_1.nii.gz'))
        #sitk.WriteTransform(new_initial_transform,path.join('/home/rl23/Desktop/data/CorrectTransformationPipeline/image_check', f'new_ini_tran{case_id}_rx45.tfm'))
        #GetImageFromArray will set orgin to 000
        np_mri=sitk.GetArrayFromImage(mri)
        np_us=sitk.GetArrayFromImage(us)
        #print('np_mri shape {}'.format(np_mri.shape))
        #print('np_us shape {}'.format(np_us.shape))
        sample4D[0, :, :, :] = np_mri
        sample4D[1, :, :, :] = np_us
        sample4D = scale_volume_std(sample4D)
        
        # mat_diff = gt_mat.dot(np.linalg.inv(base_mat))
        # target = load_func.decompose_matrix_degree(mat_diff)
        target=params_rand
        #print('target',target.shape)
        return sample4D, target,gt_trans_fn,us_center, mri_landmarks_file, us_landmarks_file

#

# ----- #
def _get_random_value(r, center, hasSign, seed=None):
    if not seed is None:
        random.seed(seed)
    
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

    euler_angle = 5
    
    angleX = _get_random_value(euler_angle, 0, signed)
    angleY = _get_random_value(euler_angle, 0, signed)
    angleZ = _get_random_value(euler_angle, 0, signed)

    translation_range = 5
    tX = _get_random_value(translation_range, 0, signed)
    tY = _get_random_value(translation_range, 0, signed)
    tZ = _get_random_value(translation_range, 0, signed)

    
    #parameters = np.asarray([0, 0, 45, 0, 0, 0])
    parameters = np.asarray([angleX, angleY, angleZ,tX, tY, tZ])
    #print('parameters',parameters)
    #arrTrans = load_func.construct_matrix_degree(parameters,
    #                                            initial_transform=base_trans_mat4x4)
    arrTrans=load_func.construct_matrix(parameters)
    #print('arrTrans',arrTrans)

    return arrTrans, parameters


def train_model(model, criterion, optimizer, scheduler, fn_save, num_epochs=25):
    since = time.time()

    lowest_loss = 2000
    lowest_TRE = 2000
    tv_hist = {'train': [], 'val': []}

    for epoch in range(num_epochs):
        #print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        global current_epoch
        current_epoch = epoch + 1

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            # print('Network is in {}...'.format(phase))

            if phase == 'train':
                
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_TRE = 0.0

            # Iterate over data.
            for inputs, labels,gt_trans_fn, us_center, mri_landmark_file, us_landmark_file in dataloaders[phase]:
                load_func.seed_reproducer(2333)
               
                labels = labels.type(torch.FloatTensor)
                #print('labels',type(labels))
                inputs = inputs.type(torch.FloatTensor)

                labels = labels.to(device)
                inputs = inputs.to(device)


                labels.require_grad = True

                optimizer.zero_grad()


                with torch.set_grad_enabled(phase == 'train'):
                    load_func.seed_reproducer(seed=2333)

                    outputs = model(inputs)
                    '''Weighted MSE loss function'''
                    loss = criterion(outputs, labels)

                    outputs = outputs.data.cpu().numpy()
                    
                    #add gt with the output to calculate the TRE
                    gt=sitk.ReadTransform(gt_trans_fn[0])
                    us_center_tuple = tuple(t.item() for t in us_center)
                    predict_trans=load_func.construct_matrix_degree_new(outputs.flatten().tolist(),initial_transform=gt,
                                                                      center=us_center_tuple)
                    ini_trans=load_func.construct_matrix_degree_new(labels.flatten().tolist(),initial_transform=gt,
                                                                      center=us_center_tuple)
                    mri_landmark=load_func.load_landmarks(mri_landmark_file[0])
                    us_landmark=load_func.load_landmarks(us_landmark_file[0])
                    evaluator=Evaluator(mri_landmark, us_landmark)
                    case_number=int(us_landmark_file[0].split('/')[-1].split('-')[0].split('Case')[-1])
                    
                    ini_=evaluator.evaluate_transform(ini_trans)
                    gt_tre=evaluator.evaluate_transform(gt)[0]
                    ini_tre=ini_[0]
                    ini_us_landmark=ini_[1]
                    batch_TRE=load_func.predict_tre(ini_us_landmark,mri_landmark,predict_trans)
                    # print('case_number',case_number)
                    # print('labels',labels)
                    # print('outputs',outputs)
                    # print('loss'    ,loss)
                    # print('gt_tre',gt_tre)
                    # print('ini_tre',ini_tre)
                    # print('batch_TRE',batch_TRE)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        #print('labels',labels)
                        #print('outputs',outputs)
                    # if phase == 'val':
                    #     print('labels',labels)
                    #     print('outputs',outputs)
                    #if epoch==num_epochs-1:
                        #if case_number==2:
                            #sitk.WriteTransform(ini_trans,'/home/rl23/Desktop/data/CorrectTransformationPipeline/val_check/ini_case2_final.tfm')

             

                running_loss += loss.data.mean() * inputs.size(0)
                #keep track of TRE of every epoch if you have the ground truth segmentation
                running_TRE += batch_TRE * inputs.size(0)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_TRE = running_TRE / dataset_sizes[phase]

            tv_hist[phase].append([epoch_loss, epoch_TRE])
            #tv_hist[phase].append(epoch_loss)
            if phase == 'train':
                scheduler.step()

            
            if phase == 'val' and epoch_loss <= lowest_loss: #loss version
            # if phase == 'val' and epoch_TRE <= lowest_TRE: #TRE version
                lowest_loss = epoch_loss
                # lowest_TRE = epoch_TRE
                best_ep = epoch
                torch.save(model.state_dict(), fn_save)
                print('**** best model updated with Loss={:.4f} ****'.format(lowest_loss))
            
                
            
            
        for param_group in optimizer.param_groups:
            print('learning rate: ',param_group['lr'])
        # print('ep {}/{}: T-loss: {:.4f}, V-loss: {:.4f}'.format(
        #     epoch + 1, num_epochs,
        #     tv_hist['train'][-1],
        #     tv_hist['val'][-1])
        # )
        #If you have the ground truth and want to keep track of TRE:
        print('ep {}/{}: T-loss: {:.4f}, V-loss: {:.4f}, T-TRE: {:.4f}, V-TRE: {:.4f}'.format(
            epoch + 1, num_epochs,
            tv_hist['train'][-1][0],
            tv_hist['val'][-1][0],
            tv_hist['train'][-1][1],
            tv_hist['val'][-1][1])
        )

    time_elapsed = time.time() - since
    print('*' * 10 + 'Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('*' * 10 + 'Lowest val loss: {:4f} at epoch {}'.format(lowest_loss, best_ep))


    return tv_hist

if __name__ == '__main__':
    

    data_dir = '/home/rl23/Desktop/data/CorrectTransformationPipeline/test_data'
    results_dir = '/home/rl23/Desktop/code/dummy/Attention-Reg/experiments/novalini_r5t5/'
    os.makedirs(results_dir, exist_ok=True)


    init_mode = args.init_mode
    network_type = args.network_type
    print('Transform initialization mode: {}'.format(init_mode))


    image_datasets = {x: MR_TRUS_4D(os.path.join(data_dir, x), init_mode)
                      for x in ['train', 'val']}
    print('image_datasets',image_datasets)
    
    #image_dataset=[train,val ,test]

#this can remain the same only change the image_dataset
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  num_workers=0)
                   for x in ['train', 'val']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    print('Number of training samples: {}'.format(dataset_sizes['train']))
    print('Number of validation samples: {}'.format(dataset_sizes['val']))

    if network_type == 'AttentionReg':
        model_ft = gens.AttentionReg()
    else:
        print('network type of <{}> is not supported, use FeatureReg instead'.format(network_type))
        model_ft = gens.FeatureReg()



    model_ft = nn.DataParallel(model_ft)
    model_ft.cuda()
    model_ft = model_ft.to(device)


    criterion = nn.MSELoss()


    lr = args.learning_rate
    print('Learning rate = {}'.format(lr))

    optimizer = optim.Adam(model_ft.parameters(), lr=lr,weight_decay=0.0001) #try removing weight decay

    # this is the learning rate that worked best for us. The network is pretty sensitive to learning rate changes.
    exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[25, 50, 100], gamma=0.3) #try gamma=0.1, milestones=[50, 150, 250]
    # exp_lr_scheduler = scheduler = CyclicLR(optimizer, 
    #                  base_lr=1e-6,   # Minimum learning rate
    #                  max_lr=1e-3,    # Maximum learning rate
    #                  step_size_up=50,  # Number of iterations to reach max_lr
    #                  mode='triangular',  # Type of cyclic pattern
    #                  cycle_momentum=False  # Set to False when using Adam
    #                 )

    now = datetime.now()
    now_str = now.strftime('%m%d-%H%M%S')
    print(now_str)


    # Ready to start
    fn_best_model = path.join(results_dir, 'Gen_{}_{}_{}_model.pth'.format(network_type, now_str, init_mode))
    print('Start training...')
    print('This model is <{}_{}_{}.pth>'.format(network_type, now_str, init_mode))
    txt_path = path.join('results/', 'training_progress_{}_{}_{}.txt'.format(network_type, now_str, init_mode))

    #count the parameters
    model_parameters = filter(lambda p: p.requires_grad, model_ft.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('params {}'.format(params))


    hist_ft = train_model(model_ft,
                          criterion,
                          optimizer,
                          exp_lr_scheduler,
                          fn_best_model,
                          num_epochs=epochs)

    fn_hist = os.path.join(results_dir, 'hist_{}_{}_{}_multi_LRgama0.3faninweightdecay0.0001_local_ini.npy'.format(net, now_str, init_mode))
    np.save(fn_hist, hist_ft)


    now = datetime.now()
    now_stamp = now.strftime('%Y-%m-%d %H:%M:%S')
    print('#' * 15 + ' Training {} completed at {} started at {}'.format(init_mode, now_stamp,now_str) + '#' * 15)
