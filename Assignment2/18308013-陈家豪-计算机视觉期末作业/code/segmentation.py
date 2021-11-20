import matplotlib.pyplot as plt # plt 用于显示图片
from PIL import Image
import numpy as np
import seaborn as sns
from seaborn.matrix import clustermap
MAX_NUM = 1000

class MyGaussianBlur():
    def __init__(self, radius = 1, sigma = 1.5):
        '''
        Description
        ----------
        高斯模糊
        
        Parameters
        ----------
        radius : int
            半径

        sigma : float
            $\sigma$值
        '''
        self.__radius = radius
        self.__sigma2 = sigma*sigma
        self.__filterLen = radius*2+1
        self.__filter = np.zeros((self.__filterLen, self.__filterLen))
        for i in range(radius*2+1):
            for j in range(radius*2+1):
                self.__filter[i,j] = self.__calcGuassian(i-radius,j-radius)
        self.__filter /= np.sum(self.__filter)
        
    
    def __calcGuassian(self,x,y):
        return np.exp(-(x*x+y*y)/(2*self.__sigma2))/(2*np.pi*self.__sigma2)

    def smooth(self, img):
        '''
        Description
        ----------
        得到img的高斯模糊矩阵
        
        Parameters
        ----------
        img : ndarray (height, width, _)
            原始图像

        sigma : float
            $\sigma$值

        Returns
        ----------
        output : ndarray (height, width, _)
            高斯模糊后的图像
        '''
        if len(img.shape)==3:
            tempImg = img
        elif len(img.shape)==2:
            tempImg = img[:,:,np.newaxis]
        paddedImg = np.pad(tempImg, ((self.__radius, self.__radius), (self.__radius, self.__radius), (0,0)), 'constant', constant_values=0)

        output = np.zeros_like(tempImg)
        filter = np.repeat(self.__filter[:,:, np.newaxis], tempImg.shape[2], axis=2)
        for i in range(output.shape[0]):
            for j in range(output.shape[1]):
                output[i,j,:] = np.sum(paddedImg[i:i+self.__filterLen, j:j+self.__filterLen, :]*filter, axis=(0,1))
        return output.reshape(img.shape)

class Segmentation():
    def __init__(self, k, min_size = 50, gaussian_radius = 1, gaussian_sigma=1.5):
        '''
        Description
        ----------
        基于图的分割
        
        Parameters
        ----------
        k : int
            修正项常数

        min_size : int
            区域像素最小数量
        
        gaussian_radius : int
            高斯模糊半径

        gaussian_sigma : float
            高斯模糊$\sigma$值
        '''
        self.k = k
        self.__minSize = min_size
        self.__gaussianRadius = gaussian_radius
        self.__gaussianSigma = gaussian_sigma
        
        
    def __getEdgeValue(self, img):
        '''
        获取八连通边的权重值字典
        '''
        blur = MyGaussianBlur(self.__gaussianRadius, self.__gaussianSigma)
        tempImg = blur.smooth(img)

        edgeDict = dict()
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if j<img.shape[1]-1:
                    edgeDict[((i,j),(i,j+1))] = self.__getPixelDiff(tempImg[i,j], tempImg[i,j+1])
                if i<img.shape[0]-1:
                    edgeDict[((i,j),(i+1,j))] = self.__getPixelDiff(tempImg[i,j], tempImg[i+1,j])
                    if j>0:
                        edgeDict[((i,j),(i+1,j))] = self.__getPixelDiff(tempImg[i,j], tempImg[i+1,j-1])
                    if j<img.shape[1]-1:
                        edgeDict[((i,j),(i+1,j+1))] = self.__getPixelDiff(tempImg[i,j], tempImg[i+1,j+1])
        return edgeDict

    def __getPixelDiff(self, pixel1, pixel2):
        '''
        计算两个像素的不相似度
        '''
        return np.sqrt(np.sum((pixel2-pixel1)**2))

    def getCluster(self, img):
        '''
        Description
        ----------
        获取图像的图分割
        
        Parameters
        ----------
        img : ndarray (height, width, 3)
            RGB像素矩阵
        
        Returns
        ----------
        clusterIndexMap : ndarray (height, width)
            分割结果矩阵，clusterIndexMap[i,j]是像素(i,j)所属区域的id

        clusterIdList: list(int)
            区域id列表
        '''
        edgeDict = self.__getEdgeValue(img)
        clusterIndexMap = np.ones((img.shape[0],img.shape[1]),dtype=int)*-1# 标号矩阵
        clusterDict = dict() # 存储各个区域的信息
        clusterId = 1
        # 初始情况下每个像素就是一个区域
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                clusterDict[clusterId] = dict()
                clusterDict[clusterId]['list'] = [(i,j)] # 区域像素坐标列表
                clusterDict[clusterId]['diff'] = self.k/len(clusterDict[clusterId]['list']) # 类内差异
                clusterIndexMap[i,j] = clusterId
                clusterId += 1
        # 排序
        edgeSortList = sorted(edgeDict.items(), key = lambda ed:(ed[1],ed[0]))

        # 合并
        for ((pixelInx1,pixelInx2),edgeValue) in edgeSortList:
            # 获取所属类标号
            clusterId1 = clusterIndexMap[pixelInx1]
            clusterId2 = clusterIndexMap[pixelInx2]
            if clusterId1==clusterId2:
                continue
            if edgeValue>min(clusterDict[clusterId1]['diff'], clusterDict[clusterId2]['diff']):
                continue
            # 合并两个区域
            clusterDict[clusterId1]['list'].extend(clusterDict[clusterId2]['list'])
            num = len(clusterDict[clusterId1]['list'])
            clusterDict[clusterId1]['diff'] = edgeValue + self.k/num
            clusterDict.pop(clusterId2)
            clusterIndexMap[clusterIndexMap==clusterId2] = clusterId1
        # 合并过小的区域
        for ((pixelInx1,pixelInx2),edgeValue) in edgeSortList:
            clusterId1 = clusterIndexMap[pixelInx1]
            clusterId2 = clusterIndexMap[pixelInx2]
            if clusterId1==clusterId2:
                continue
            if len(clusterDict[clusterId1]['list'])>=self.__minSize and len(clusterDict[clusterId2]['list'])>=self.__minSize:
                continue
            clusterDict[clusterId1]['list'].extend(clusterDict[clusterId2]['list'])
            num = len(clusterDict[clusterId1]['list'])
            clusterDict[clusterId1]['diff'] = edgeValue + self.k/num
            clusterDict.pop(clusterId2)
            clusterIndexMap[clusterIndexMap==clusterId2] = clusterId1
        
        return clusterIndexMap, list(clusterDict.keys())

    def getColorImg(self, img, clusterIndexMap, clusterIndexList):
        '''
        获取标记分割的图像，不同区域用不同颜色表示
        '''
        newImgArray = img.copy()
        colorList = list()
        for i in clusterIndexList:
            color = np.random.randint(256,size=(3))
            colorList.append(color)
            newImgArray[clusterIndexMap==i] = color
        return newImgArray

    def getSuperPixelImg(self, img, clusterIndexMap, clusterIndexList):
        '''
        获取标记分割的图像，不同区域用红色边包围
        '''
        newImgArray = img.copy()
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                left = max(j-1,0)
                right = min(img.shape[1]-1,j+1)
                top = max(0, i-1)
                bottom = min(i+1, img.shape[0]-1)
                if np.sum((clusterIndexMap[top:bottom+1,left:right+1]!=clusterIndexMap[i,j]).astype(int))>0:
                    newImgArray[i,j,:] = np.array([255,0,0])

        return newImgArray

    def getMaskedImg(self, img, mask, clusterIndexMap, clusterIndexList):
        '''
        获取img新的mask
        '''
        newMask = mask.copy()
        newMask[mask>0] = 1
        newImgArray = np.zeros_like(img,dtype=int)
        labelList = list()# 不同区域的标签
        for i in clusterIndexList:
            num = np.sum((clusterIndexMap==i).astype(int))
            if num/2<=np.sum(newMask[clusterIndexMap==i])/3:
                # 属于前景
                newImgArray[clusterIndexMap==i] = 255
                labelList.append(1)
            else:
                # 属于背景
                labelList.append(0)
        return newImgArray, labelList

    def calcIoU(self, img, mask, clusterIndexMap, clusterIndexList):
        '''
        计算IoU值
        '''
        newMask,_ = self.getMaskedImg(img, mask, clusterIndexMap, clusterIndexList)
        R1 = newMask[:,:,0]
        R2 = mask[:,:,0].copy()
        R1[R1>0] = 1
        R2[R2>0] = 1
        intersection = np.sum(R1*R2)
        union = np.sum(R1+R2)-intersection
        return intersection/union

def main():
    imgList = list()
    maskList = list()
    usedIdList = list()
    IoUList = list()
    id = 13
    while id<MAX_NUM:
        usedIdList.append(id)
        id += 100
    for id in usedIdList:
        img = Image.open('imgs/{}.png'.format(id))
        imgList.append(np.asarray(img))
        img = Image.open('gt/{}.png'.format(id))
        maskList.append(np.asarray(img))

    model = Segmentation(k=200, gaussian_radius=1, gaussian_sigma=0.8)
    for i in range(len(usedIdList)):
        model.k = 150
        clusterIndexMap, clusterIndexList = model.getCluster(imgList[i])
        while len(clusterIndexList)<50 or len(clusterIndexList)>70:
            if len(clusterIndexList)<50:
                model.k = int(model.k*0.8)
                if model.k<=0:
                    break
            else:
                model.k = int(model.k*1.2)
                if model.k>=1000:
                    break
            clusterIndexMap, clusterIndexList = model.getCluster(imgList[i])
        newMask,_ = model.getMaskedImg(imgList[i], maskList[i], clusterIndexMap, clusterIndexList)
        Image.fromarray(np.uint8(newMask)).save('new_gt/{}_new_mask.png'.format(usedIdList[i]))
        newImg = model.getSuperPixelImg(imgList[i], clusterIndexMap, clusterIndexList)
        Image.fromarray(np.uint8(newImg)).save('new_imgs/{}_segmentation.png'.format(usedIdList[i]))
        IoUList.append(model.calcIoU(imgList[i], maskList[i], clusterIndexMap, clusterIndexList))
        print("#{} IoU:{}".format(usedIdList[i], IoUList[i]))
    
    print("average IoU: {}".format(np.sum(np.array(IoUList))/len(IoUList)))

if __name__=='__main__':
    main()