"""
created by haoran
time:10/14
description:
输入数据的准备，输入数据来自不同的方法制作的数据
input: 各种类型数据集的路径
output: 输出的是一个train list & gt list
"""
import traceback
import numpy
from PIL import Image
import matplotlib.pyplot as plt
import sys, os, shutil
import random
class MixData():
    def __init__(self):
        # src_path_list = [ '/media/liu/File/10月数据准备/10月12日实验数据/splicing/tamper_result_320',
            # '/media/liu/File/10月数据准备/10月12日实验数据/cm/test_dataset_train_percent_0.80@8_20']
                         # '/media/liu/File/10月数据准备/10月12日实验数据/negative/src',
                         # '/media/liu/File/10月数据准备/10月12日实验数据/casia/src',
        src_path_list = [#'/media/liu/File/10月数据准备/10月12日实验数据/cm/test_dataset_train_percent_0.80@8_20',
                         #'/media/liu/File/10月数据准备/10月12日实验数据/negative/src',
                         # '/media/liu/File/10月数据准备/10月12日实验数据/cm/test_dataset_train_percent_0.80@8_20',
                        # '/media/liu/File/10月数据准备/10月12日实验数据/splicing/tamper_result_320',
                        # '/media/liu/File/10月数据准备/10月12日实验数据/casia/src'
                         ]
        gt_path_list = []
        self.src_path_list = src_path_list

        self.sp_gt_path = '/media/liu/File/10月数据准备/10月12日实验数据/splicing/ground_truth_result_320'
        self.cm_gt_path = '/media/liu/File/10月数据准备/10月12日实验数据/cm/test_gt_train_percent_0.80@8_20'
        self.negative_gt_path = '/media/liu/File/10月数据准备/10月12日实验数据/negative/gt'
        self.casia_gt_path = '/media/liu/File/10月数据准备/10月12日实验数据/casia/gt'

    def gen_dataset(self):
        """
        通过输入的src & gt的路径生成train_list 列表
        并通过check方法，检查是否有误
        :return:
        """
        dataset_type_num = len(self.src_path_list)
        train_list = []
        gt_list = []
        unmatched_list = []
        # 首先开始遍历不同类型的数据集路径
        for index1, item1 in enumerate(self.src_path_list):
            for index2,item2 in enumerate(os.listdir(item1)):
                t_img_path = os.path.join(item1, item2)
                t_gt_path = MixData.__switch_case(self, t_img_path)
                if t_gt_path != '':
                    train_list.append(t_img_path)
                    gt_list.append(t_gt_path)
                else:
                    print(t_gt_path, t_gt_path,'unmatched')
                    unmatched_list.append([t_img_path,t_gt_path])
                    print('The process: %d/%d : %d/%d'%(index1+1, len(self.src_path_list), index2+1,len((os.listdir(item1)))))
        print('The number of unmatched data is :', len(unmatched_list))
        print('The unmatched list is : ',unmatched_list)


        # if MixData.__check(self,train_list=, gt_list=):
        #     pass
        # else:
        #     print('check error, please redesign')
        #     traceback.print_exc()
        #     sys.exit()

        return train_list, gt_list


    def __check(self, train_list, gt_list):
        """
        检查train_list 和 gt_list 是否有问题
        :return:
        """
        pass

    def __switch_case(self, path):
        """
        针对不同类型的数据集做处理
        :return: 返回一个路径，这个路径是path 所对应的gt路径，并且需要检查该路径是否存在
        """
        # 0 判断路径的合法性
        if os.path.exists(path):
            pass
        else:
            print('The path :', path, 'does not exist')
            return ''
        # 1 分析属于何种类型
        # there are
        # 1.  sp generate data
        # 2. cm generate data
        # 3. negative data
        # 4. CASIA data

        sp_type = ['Sp']
        cm_type = ['Default','poisson']
        negative_type = ['negative']
        CASIA_type = ['Tp']
        type= []
        name = path.split('/')[-1]

        for sp_flag in sp_type:
            if sp_flag in name[:2]:
               type.append('sp')
               break

        for cm_flag in cm_type:
            if cm_flag in name[:7]:
                type.append('cm')
                break

        for negative_flag in negative_type:
            if negative_flag in name:
                type.append('negative')
                break

        for CASIA_flag in CASIA_type:
            if CASIA_flag in name[:2]:
                type.append('casia')
                break

        # 判断正确性
        if len(type) != 1:
            print('The type len is ', len(type))
            return ''

        if type[0] == 'sp':
            gt_path = name.replace('Default','Gt').replace('.jpg','.bmp').replace('.png', '.bmp').replace('poisson','Gt')
            gt_path = os.path.join(self.sp_gt_path, gt_path)
            pass
        elif type[0] == 'cm':
            gt_path = name.replace('Default', 'Gt').replace('.jpg','.bmp').replace('.png', '.bmp').replace('poisson','Gt')
            gt_path = os.path.join(self.cm_gt_path, gt_path)
            pass
        elif type[0] == 'negative':
            gt_path = 'negative_gt.bmp'
            gt_path = os.path.join(self.negative_gt_path, gt_path)
            pass
        elif type[0] == 'casia':
            gt_path = name.split('.')[0] + '_gt' + '.png'
            gt_path = os.path.join(self.casia_gt_path, gt_path)
            pass
        else:
            traceback.print_exc()
            print('Error')
            sys.exit()
        # 判断gt是否存在
        if os.path.exists(gt_path):
            pass
        else:
            return ''

        return gt_path
if __name__ == '__main__':
    MixData().gen_dataset()