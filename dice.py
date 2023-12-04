import cv2
import os
import numpy as np
from medpy import metric

def dice(path, GTpath):
    '''
    root 所指的是当前正在遍历的这个文件夹的本身的地址
    dirs 是一个 list ，内容是该文件夹中所有的目录的名字(不包括子目录)
    files 同样是 list , 内容是该文件夹中所有的文件名(不包括子目录)
    '''
    jiaosum = 0
    bingsum = 0
    d = []
    v = []
    r = []
    a = []
    lowlist = []
    for root, dirs, files in os.walk(path):
        for name in files:
            mask_dir = GTpath + name
            mask = cv2.imread(mask_dir, 0)
            pred_dir = path + name
            pred = cv2.imread(pred_dir, 0)
            # pred = cv2.threshold(pred, 199, 255, cv2.THRESH_BINARY)[1]

            fmask = mask.flatten()
            fpred = pred.flatten()

            if 255 in fmask or 255 in fpred:
                eps = 1
                fmask = fmask / 255
                fpred = fpred / 255

                jiao = np.sum(fmask * fpred)
                jiaosum += jiao
                bing = np.sum(fmask) + np.sum(fpred)
                bingsum += bing
                dice = ((2. * jiao) / (bing + eps))
                if 1 in fmask:
                    voe = 2*abs((np.sum(fpred)-np.sum(fmask)))/bing
                    if voe < 1:
                        v.append(voe)
                    else:
                        lowlist.append(name)

                    rvd = (np.sum(fpred)/np.sum(fmask))-1
                    r.append(rvd)

                    # asd = metric.binary.asd(pred, mask)
                    # a.append(asd)


                d.append(dice)
                if dice < 0.9 and np.sum(fmask) > 1400:
                    lowlist.append(name)
                    # analpath = "D:/DataSet/LiTS/2.0_test/analresult100/" + name
                    # cv2.imwrite(analpath, pred)

    d = list(d)
    # v = list(v)
    # r = list(r)
    # a = list(a)
    aveDICE = sum(d) / len(d)
    # aveVOE = sum(v) / len(v)
    # aveRVD = sum(r) / len(r)
    # aveASD = sum(a) / len(a)

    GlobalDice = jiaosum*2/bingsum
    print("GlobalDice:", GlobalDice,
          "Dice", aveDICE)
          # "VOE:", aveVOE,
          # "RVD:", aveRVD)
    print("worst list:", lowlist)


# resultpath = "D:/DataSet/LiTS/2.0_test/result100/12/"
# GTpath = "D:/DataSet/LiTS/2.0_test/mask/12/"
# resultpath = "D:/DataSet/Company/test/120output/"
# GTpath = "D:/DataSet/Company/test/liverlabel/"
# resultpath = "D:/DataSet/Company/company_sj/result_traUnet/result2/"
# GTpath = "D:/DataSet/Company/company_sj/Liver_testing_images/mask3/"
resultpath = "D:\\DataSet\\3Dircadb\\s1+s2Result\\1.4\\"
GTpath = "D:\DataSet\\3Dircadb\\next_ct\\testgt\\1.4\\"
dice(resultpath, GTpath)
