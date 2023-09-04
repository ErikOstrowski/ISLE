import os
import cv2
import math
import pandas as pd
import numpy as np
from PIL import Image
import multiprocessing
import argparse

# EXECUTE THIS PROGRAM INSIDE THE PUZZLECAM FOLDER!

parser = argparse.ArgumentParser()
parser.add_argument('--experiment_name', default='resnet50@seed=0@nesterov@train@bg=0.20@scale=0.5,1.0,1.5,2.0@png', type=str)
parser.add_argument("--domain", default='train', type=str)
parser.add_argument("--threshold", default=None, type=float)

parser.add_argument("--predict_dir", default='', type=str)
parser.add_argument('--gt_dir', default='/srv/data/eostrowski/Dataset/VOC2012/SegmentationClass', type=str)

parser.add_argument('--logfile', default='',type=str)
parser.add_argument('--comment', default='', type=str)

parser.add_argument('--mode', default='npy', type=str) # png
parser.add_argument('--max_th', default=0.50, type=float)

args = parser.parse_args()

predict_folder = '/PATH/TO/PUZZLECAM/'
predict_folder2= '/PATH/TO/CLIMS/'
predict_folderdrs= '/PATH/TO/DRS/'
predict_folderpmm= '/PATH/TO/PMM/'
save_path = '/ISLE/SAVE/PATH/'
gt_folder = args.gt_dir

args.list = './data/' + args.domain + '.txt'
args.predict_dir = predict_folder

categories = ['background', 
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 
    'bus', 'car', 'cat', 'chair', 'cow', 
    'diningtable', 'dog', 'horse', 'motorbike', 'person', 
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
num_cls = len(categories)

def compare(start,step,TP,P,T, name_list):
    for idx in range(start,len(name_list),step):
        name = name_list[idx]

        if os.path.isfile(predict_folder + name + '.npy'):
            predict_dict = np.load(os.path.join(predict_folder, name + '.npy'), allow_pickle=True).item()
            predict_dict2 = np.load(os.path.join(predict_folder2, name + '.npy'), allow_pickle=True).item()
            predict_dictdrs = np.load(os.path.join(predict_folderdrs, name + '.npy'), allow_pickle=True).item()
            predict_dictpmm = np.load(os.path.join(predict_folderpmm, name + '.npy'), allow_pickle=True).item()

            if 'hr_cam' in predict_dict.keys():
                cams = predict_dict['hr_cam']
                cams = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=args.threshold)
                
                cams2 = predict_dict2['hr_cam']
                cams2 = np.pad(cams2, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=args.threshold)
                
                camsdrs = predict_dictdrs['hr_cam']
                camsdrs = np.pad(camsdrs, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=args.threshold)

                camspmm = predict_dictpmm['hr_cam']
                camspmm = np.pad(camspmm, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=args.threshold)


            elif 'rw' in predict_dict.keys():
                cams = predict_dict['rw']
                cams = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=args.threshold)
            
                cams2 = predict_dict2['rw']
                cams2 = np.pad(cams2, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=args.threshold)

                camsdrs = predict_dictdrs['rw']
                camsdrs = np.pad(camsdrs, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=args.threshold)
                
                camspmm = predict_dictpmm['rw']
                camspmm = np.pad(camspmm, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=args.threshold)

            keys = predict_dict['keys']
            keys2 = predict_dict2['keys']
            keysdrs = predict_dictdrs['keys']
            keyspmm = predict_dictpmm['keys']

            predict = keys[np.argmax(cams, axis=0)]
            predict2 = keys2[np.argmax(cams2, axis=0)]

            predictdrs = keysdrs[np.argmax(camsdrs, axis=0)]
            predictpmm = keyspmm[np.argmax(camspmm, axis=0)]

        else:
            predict = np.array(Image.open(predict_folder + name + '.png'))
            predict2 = np.array(Image.open(predict_folder2 + name + '.png'))
            predictdrs = np.array(Image.open(predict_folderdrs + name + '.png'))
            predictpmm = np.array(Image.open(predict_folderpmm + name + '.png'))



        saver = np.zeros(predict.shape)
        
        # aero
        #aero = predict[predict==1]
        #saver = saver + 
        aero = np.array(predict, copy=True)  #predict2
        aero[(aero!=1)]=0

        saver =	saver +	aero #predict2[predict2==1]

        bike =  np.array(predict2, copy=True)  #predict
        bike[(bike!=2)]=0
       	saver = saver + bike#predict[predict==2]

        bird = np.array(predict, copy=True)  #predict2
        bird[(bird!=3)]=0
       	saver = saver + bird #predict2[predict2==3]

        boat = np.array(predict2, copy=True)  #predict
        boat[(boat!=4)]=0
       	saver = saver + boat #predict[predict==4]

        bottle = np.array(predictdrs, copy=True)  #predict2
        bottle[(bottle!=5)]=0
       	saver = saver + bottle #predict2[predict2==5]

        bus = np.array(predictpmm, copy=True)  #predict
        bus[(bus!=6)]=0
       	saver = saver + bus #predict[predict==6]

        car = np.array(predictpmm, copy=True)  #predict2
        car[(car!=7)]=0
        saver = saver + car #predict2[predict2==7]

        cat = np.array(predict, copy=True)  #predict2
        cat[(cat!=8)]=0
        saver = saver + cat#predict2[predict2==8]

        chair = np.array(predictdrs, copy=True)  #predict2
        chair[(chair!=9)]=0
        saver = saver +chair #predict2[predict2==9]

        cow = np.array(predict, copy=True)  #predict2
        cow[(cow!=10)]=0
        saver = saver + cow#predict2[predict2==10]

        table = np.array(predict2, copy=True)  #predict
        table[(table!=11)]=0
       	saver = saver + table #predict[predict==11]

        dog = np.array(predict, copy=True)  #predict2
        dog[(dog!=12)]=0
        saver = saver + dog #predict2[predict2==12]

        horse = np.array(predict, copy=True)  #predict2
        horse[(horse!=13)]=0
        saver = saver + horse #predict2[predict2==13]

        motor = np.array(predict2, copy=True)  #predict
        motor[(motor!=14)]=0
        saver = saver + motor#predict[predict==14]

        person = np.array(predict2, copy=True)  #predict
        person[(person!=15)]=0
        saver = saver + person#predict[predict==15]

        plant = np.array(predict, copy=True)  #predict2
        plant[(plant!=16)]=0
       	saver = saver + plant#predict2[predict2==16]

        sheep = np.array(predict, copy=True)  #predict2
        sheep[(sheep!=17)]=0
        saver = saver + sheep #predict2[predict2==17]

        sofa = np.array(predict2, copy=True)  #predict
        sofa[(sofa!=18)]=0
        saver = saver + sofa #predict[predict==18]

        train = np.array(predictdrs, copy=True)  #predict
        train[(train!=19)]=0
        saver = saver + train #predict[predict==19]

        tv = np.array(predict2, copy=True)  #predict
        tv[(tv!=20)]=0
        saver = saver + tv #predict[predict==20]
        

        saver[saver>20]=0
        saver_file = os.path.join(save_path,name)
        #print(saver.shape)
        #np.save(saver_file,saver)
        im = Image.fromarray(saver)
        im = im.convert('L')
        im.save(save_path + name + '.png')


        gt_file = os.path.join(gt_folder,'%s.png'%name)
        gt = np.array(Image.open(gt_file))
        #print(f'UNIQUELO LABELS IN PERDICTION:      {np.unique(predict)} \n')
        cal = gt<255
        mask = (predict==gt) * cal
        for i in range(num_cls):
            P[i].acquire()
            P[i].value += np.sum((predict==i)*cal)
            P[i].release()
            T[i].acquire()
            T[i].value += np.sum((gt==i)*cal)
            T[i].release()
            TP[i].acquire()
            TP[i].value += np.sum((gt==i)*mask)
            TP[i].release()
            
def do_python_eval(predict_folder, gt_folder, name_list, num_cls=21):
    TP = []
    P = []
    T = []
    for i in range(num_cls):
        TP.append(multiprocessing.Value('i', 0, lock=True))
        P.append(multiprocessing.Value('i', 0, lock=True))
        T.append(multiprocessing.Value('i', 0, lock=True))
    
    p_list = []
    for i in range(1):
        p = multiprocessing.Process(target=compare, args=(i,1,TP,P,T, name_list))
        p.start()
        p_list.append(p)
    for p in p_list:
        p.join()
    IoU = []
    T_TP = []
    P_TP = []
    FP_ALL = []
    FN_ALL = [] 
    for i in range(num_cls):
        IoU.append(TP[i].value/(T[i].value+P[i].value-TP[i].value+1e-10))
        T_TP.append(T[i].value/(TP[i].value+1e-10))
        P_TP.append(P[i].value/(TP[i].value+1e-10))
        FP_ALL.append((P[i].value-TP[i].value)/(T[i].value + P[i].value - TP[i].value + 1e-10))
        FN_ALL.append((T[i].value-TP[i].value)/(T[i].value + P[i].value - TP[i].value + 1e-10))

    loglist = {}
    for i in range(num_cls):
         if i%2 != 1:
             print('%11s:%7.3f%%'%(categories[i],IoU[i]*100),end='\t')
         else:
             print('%11s:%7.3f%%'%(categories[i],IoU[i]*100))
         loglist[categories[i]] = IoU[i] * 100
    
    miou = np.mean(np.array(IoU))
    t_tp = np.mean(np.array(T_TP)[1:])
    p_tp = np.mean(np.array(P_TP)[1:])
    fp_all = np.mean(np.array(FP_ALL)[1:])
    fn_all = np.mean(np.array(FN_ALL)[1:])
    miou_foreground = np.mean(np.array(IoU)[1:])
    # print('\n======================================================')
    # print('%11s:%7.3f%%'%('mIoU',miou*100))
    # print('%11s:%7.3f'%('T/TP',t_tp))
    # print('%11s:%7.3f'%('P/TP',p_tp))
    # print('%11s:%7.3f'%('FP/ALL',fp_all))
    # print('%11s:%7.3f'%('FN/ALL',fn_all))
    # print('%11s:%7.3f'%('miou_foreground',miou_foreground))
    loglist['mIoU'] = miou * 100
    loglist['t_tp'] = t_tp
    loglist['p_tp'] = p_tp
    loglist['fp_all'] = fp_all
    loglist['fn_all'] = fn_all
    loglist['miou_foreground'] = miou_foreground 
    return loglist

if __name__ == '__main__':
    df = pd.read_csv(args.list, names=['filename'])
    name_list = df['filename'].values

    if args.mode == 'png':
        loglist = do_python_eval(args.predict_dir, args.gt_dir, name_list, 21)
        print('mIoU={:.3f}%, FP={:.4f}, FN={:.4f}'.format(loglist['mIoU'], loglist['fp_all'], loglist['fn_all']))
    elif args.mode == 'rw':
        th_list = np.arange(0.05, args.max_th, 0.05).tolist()

        over_activation = 1.60
        under_activation = 0.60
        
        mIoU_list = []
        FP_list = []

        for th in th_list:
            args.threshold = th
            loglist = do_python_eval(args.predict_dir, args.gt_dir, name_list, 21)

            mIoU, FP = loglist['mIoU'], loglist['fp_all']

            print('Th={:.2f}, mIoU={:.3f}%, FP={:.4f}'.format(th, mIoU, FP))

            FP_list.append(FP)
            mIoU_list.append(mIoU)
        
        best_index = np.argmax(mIoU_list)
        best_th = th_list[best_index]
        best_mIoU = mIoU_list[best_index]
        best_FP = FP_list[best_index]

        over_FP = best_FP * over_activation
        under_FP = best_FP * under_activation

        print('Over FP : {:.4f}, Under FP : {:.4f}'.format(over_FP, under_FP))

        over_loss_list = [np.abs(FP - over_FP) for FP in FP_list]
        under_loss_list = [np.abs(FP - under_FP) for FP in FP_list]

        over_index = np.argmin(over_loss_list)
        over_th = th_list[over_index]
        over_mIoU = mIoU_list[over_index]
        over_FP = FP_list[over_index]

        under_index = np.argmin(under_loss_list)
        under_th = th_list[under_index]
        under_mIoU = mIoU_list[under_index]
        under_FP = FP_list[under_index]
        
        print('Best Th={:.2f}, mIoU={:.3f}%, FP={:.4f}'.format(best_th, best_mIoU, best_FP))
        print('Over Th={:.2f}, mIoU={:.3f}%, FP={:.4f}'.format(over_th, over_mIoU, over_FP))
        print('Under Th={:.2f}, mIoU={:.3f}%, FP={:.4f}'.format(under_th, under_mIoU, under_FP))
    else:
        if args.threshold is None:
            th_list = np.arange(0.05, 0.80, 0.05).tolist()
            
            best_th = 0
            best_mIoU = 0

            for th in th_list:
                args.threshold = th
                loglist = do_python_eval(args.predict_dir, args.gt_dir, name_list, 21)
                print('Th={:.2f}, mIoU={:.3f}%, FP={:.4f}, FN={:.4f}'.format(args.threshold, loglist['mIoU'], loglist['fp_all'], loglist['fn_all']))

                if loglist['mIoU'] > best_mIoU:
                    best_th = th
                    best_mIoU = loglist['mIoU']
            
            print('Best Th={:.2f}, mIoU={:.3f}%'.format(best_th, best_mIoU))
        else:
            loglist = do_python_eval(args.predict_dir, args.gt_dir, name_list, 21)
            print('Th={:.2f}, mIoU={:.3f}%, FP={:.4f}, FN={:.4f}'.format(args.threshold, loglist['mIoU'], loglist['fp_all'], loglist['fn_all']))

