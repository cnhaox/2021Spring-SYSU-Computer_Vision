import matplotlib.pyplot as plt # plt 用于显示图片
from PIL import Image
import numpy as np
from tqdm import tqdm
import imageio
MAX_NUM = 1000

class SeamCarving():
    def __getDynamicEnergyMap(self, img, mask):
        '''
        Description
        ----------
        获取动态规划能量图和路径图
        
        Parameters
        ----------
        img : ndarray (height, width)
            灰度图

        mask : ndarray (height, width)
            蒙版
        
        Returns
        ----------
        dynamicEnergyMap : ndarray (height, width)
            动态规划能量图
        
        routeMap : ndarray (height, width)
            seam路径图
        '''
        height, width = img.shape
        # 用蒙版处理图像
        img = img*mask
        # 扩充0，便于后续操作
        paddingImg = np.pad(img, ((1,1),(1,1)), 'constant', constant_values=0)
        dynamicEnergyMap = np.zeros((height+2, width+2))
        routeMap = np.ones((height, width), dtype=int)*-1
        for hIndex in range(height):
            M = np.zeros((width, 3))
            M[:, 0] = dynamicEnergyMap[hIndex, :-2]+np.abs(paddingImg[hIndex][1:-1]-paddingImg[hIndex+1][:-2])
            M[:, 1] = dynamicEnergyMap[hIndex, 1:-1]
            M[:, 2] = dynamicEnergyMap[hIndex, 2:]+np.abs(paddingImg[hIndex][1:-1]-paddingImg[hIndex+1][2:])
            colEnergy = np.abs(paddingImg[hIndex+1][2:]-paddingImg[hIndex+1][:-2])
            dynamicEnergyMap[hIndex+1, 1:-1] = np.min(M, axis=1)
            routeMap[hIndex] = np.argmin(M, axis=1)-1
            # 像素(i,0)只能选择(i-1,0)和(i-1,1)
            if M[0,1]<=M[0,2]:
                routeMap[hIndex,0] = 0
                dynamicEnergyMap[hIndex+1,1] = M[0,1]
            else:
                routeMap[hIndex,0] = 1
                dynamicEnergyMap[hIndex+1,1] = M[0,2]
            # 像素(i,-1)只能选择(i-1,-1)和(i-1,-2)
            if M[-1,0]<=M[-1,1]:
                routeMap[hIndex,-1] = -1
                dynamicEnergyMap[hIndex+1,-1] = M[-1,0]
            else:
                routeMap[hIndex,-1] = 0
                dynamicEnergyMap[hIndex+1,-1] = M[-1,1]
            dynamicEnergyMap[hIndex+1,1:-1] += colEnergy

        return dynamicEnergyMap[1:-1,1:-1], routeMap

    def __getSeam(self, routeMap, index):
        '''
        Description
        ----------
        通过index获取该像素对应的seam
        
        Parameters
        ----------
        routeMap : ndarray (height, width)
            seam路径图

        index : int
            最后一行像素的索引
        
        Returns
        ----------
        seamIndex : list(int)
            seam所含像素的位置索引，第i个像素坐标为(i,seamIndex[i])
        '''
        seamIndex = np.zeros(routeMap.shape[0], dtype=int)
        seamIndex[seamIndex.size-1] = index
        for i in range(routeMap.shape[0]-2,-1,-1):
            seamIndex[i] = seamIndex[i+1]+routeMap[i+1, seamIndex[i+1]]
        return seamIndex

    def __removeSeam(self, img, seamIndex):
        '''
        Description
        ----------
        去除img中的seam
        
        Parameters
        ----------
        img : ndarray (height, width, _)
            像素矩阵

        seamIndex : list(int)
            seam所含像素的位置索引，第i个像素坐标为(i,seamIndex[i])
        
        Returns
        ----------
        newImg : ndarray (height-1, width, _)
            新的像素矩阵
        '''
        isRetain = np.ones((img.shape[0], img.shape[1]), dtype=bool)
        isRetain[seamIndex[:,None]==np.arange(img.shape[1])] = False
        if len(img.shape)==3:
            newImg = img[isRetain].reshape((img.shape[0], img.shape[1]-1, img.shape[2]))
        else:
            newImg = img[isRetain].reshape((img.shape[0], img.shape[1]-1))
        return newImg

    def __markSeam(self, rgbImg, seamIndex):
        '''
        Description
        ----------
        标记rgbImg中的seam为黑色
        
        Parameters
        ----------
        img : ndarray (height, width, 3)
            RGB像素矩阵

        seamIndex : list(int)
            seam所含像素的位置索引，第i个像素坐标为(i,seamIndex[i])
        
        Returns
        ----------
        newImg : ndarray (height, width, 3)
            新的RGB像素矩阵
        '''
        isMark = np.zeros((rgbImg.shape[0], rgbImg.shape[1]), dtype=bool)
        isMark[seamIndex[:,None]==np.arange(rgbImg.shape[1])] = True
        newImg = rgbImg.copy()
        newImg[isMark] = 0
        return newImg

    def changeShape(self, img, new_shape, mask=None, file_path="process", id=0):
        '''
        Description
        ----------
        改变img的尺寸
        
        Parameters
        ----------
        img : ndarray (height, width, 3)
            原始图像

        new_shape: tuple(new_height, new_width)
            新尺寸

        mask : ndarray (height, width)
            蒙版

        file_path : str
            保存文件夹路径
        
        id : int
            图像id
        '''
        # 确保新尺寸小于原尺寸
        assert len(new_shape)==2
        assert img.shape[0]>=new_shape[0] and img.shape[1]>=new_shape[1]
        # 修改蒙版
        if mask is None:
            newMask = np.ones((img.shape[0], img.shape[1]))
        else:
            newMask = mask.copy()
            newMask[newMask<0.5] = 0.2
        # 计算缩小的高度和宽度
        reducedHeight = img.shape[0]-new_shape[0]
        reducedWidth = img.shape[1]-new_shape[1]
        # 获取灰度图
        grayImg = np.asarray(Image.fromarray(np.uint8(img)).convert('L')).copy().astype(int)
        # 获取像素索引图，便于后续标记seam
        indexMap = np.zeros((img.shape[0],img.shape[1],2), dtype=int)
        for i in range(img.shape[0]):
            indexMap[i,:,0] = i
            indexMap[i,:,1] = range(img.shape[1])
        grayImgLists = list()# 灰度图状态
        maskLists = list()# mask状态
        imgLists = list()# 图像状态
        indexLists = list()# 图像索引矩阵状态
        seamLists = list()# 每个状态操作的seam
        costLists = list()# 每个状态的最小cost
        chosenLists = list()# 当前状态的上一个状态(对于[i,j]，0:[i,j-1]; 1:[i-1,j])
        # 动态规划
        for hIndex in tqdm(range(reducedHeight+1)):
            grayImgLists.append(list())
            maskLists.append(list())
            imgLists.append(list())
            indexLists.append(list())
            seamLists.append(list())
            costLists.append(list())
            chosenLists.append(list())
            for wIndex in range(reducedWidth+1):
                if wIndex!=0:
                    # 获取垂直seam
                    dynamicEnergyMap, routeMap = self.__getDynamicEnergyMap(grayImgLists[-1][wIndex-1], maskLists[-1][wIndex-1])
                    colSeamIndex = self.__getSeam(routeMap, np.argmin(dynamicEnergyMap[-1,:]))
                    colCost = costLists[-1][wIndex-1]+np.min(dynamicEnergyMap[-1,:])
                if hIndex!=0:
                    # 获取水平seam时，将相关矩阵转置后、通过获取垂直seam的方式获取
                    tranGrayImg = np.transpose(grayImgLists[-2][wIndex], (1,0))
                    tranImg = np.transpose(imgLists[-2][wIndex], (1,0,2))
                    tranMask = np.transpose(maskLists[-2][wIndex], (1,0))
                    tranIndexMap = np.transpose(indexLists[-2][wIndex], (1,0,2))
                    dynamicEnergyMap, routeMap = self.__getDynamicEnergyMap(tranGrayImg, tranMask)
                    rowSeamIndex = self.__getSeam(routeMap, np.argmin(dynamicEnergyMap[-1,:]))
                    rowCost = costLists[-2][wIndex]+np.min(dynamicEnergyMap[-1,:])
                if wIndex==0 and hIndex==0:
                    # 起始写入原始数据
                    grayImgLists[-1].append(grayImg)
                    maskLists[-1].append(newMask)
                    imgLists[-1].append(img)
                    indexLists[-1].append(indexMap)
                    seamLists[-1].append(None)
                    costLists[-1].append(0.0)
                    chosenLists[-1].append(-1)
                elif hIndex==0:
                    # 只有垂直seam
                    grayImgLists[-1].append(self.__removeSeam(grayImgLists[-1][-1], colSeamIndex))
                    maskLists[-1].append(self.__removeSeam(maskLists[-1][-1], colSeamIndex))
                    imgLists[-1].append(self.__removeSeam(imgLists[-1][-1], colSeamIndex))
                    indexLists[-1].append(self.__removeSeam(indexLists[-1][-1], colSeamIndex))
                    seamLists[-1].append(colSeamIndex)
                    costLists[-1].append(colCost)
                    chosenLists[-1].append(1)
                elif wIndex==0:
                    # 只有水平seam
                    grayImgLists[-1].append(np.transpose(self.__removeSeam(tranGrayImg, rowSeamIndex), (1,0)))
                    maskLists[-1].append(np.transpose(self.__removeSeam(tranMask, rowSeamIndex), (1,0)))
                    imgLists[-1].append(np.transpose(self.__removeSeam(tranImg, rowSeamIndex), (1,0,2)))
                    indexLists[-1].append(np.transpose(self.__removeSeam(tranIndexMap, rowSeamIndex), (1,0,2)))
                    seamLists[-1].append(rowSeamIndex)
                    costLists[-1].append(rowCost)
                    chosenLists[-1].append(0)
                else:
                    # 比较cost
                    if rowCost>colCost:
                        grayImgLists[-1].append(self.__removeSeam(grayImgLists[-1][-1], colSeamIndex))
                        maskLists[-1].append(self.__removeSeam(maskLists[-1][-1], colSeamIndex))
                        imgLists[-1].append(self.__removeSeam(imgLists[-1][-1], colSeamIndex))
                        indexLists[-1].append(self.__removeSeam(indexLists[-1][-1], colSeamIndex))
                        seamLists[-1].append(colSeamIndex)
                        costLists[-1].append(colCost)
                        chosenLists[-1].append(1)
                    else:
                        grayImgLists[-1].append(np.transpose(self.__removeSeam(tranGrayImg, rowSeamIndex), (1,0)))
                        maskLists[-1].append(np.transpose(self.__removeSeam(tranMask, rowSeamIndex), (1,0)))
                        imgLists[-1].append(np.transpose(self.__removeSeam(tranImg, rowSeamIndex), (1,0,2)))
                        indexLists[-1].append(np.transpose(self.__removeSeam(tranIndexMap, rowSeamIndex), (1,0,2)))
                        seamLists[-1].append(rowSeamIndex)
                        costLists[-1].append(rowCost)
                        chosenLists[-1].append(0)
                    
        # 输出最终图像
        Image.fromarray(np.uint8(imgLists[-1][-1])).save(file_path+"/"+str(id)+"_final.png")
        # 标记所有seam，输出对应图像
        isMark = np.ones((img.shape[0], img.shape[1]), dtype=bool)
        for i in range(indexLists[-1][-1].shape[0]):
            for j in range(indexLists[-1][-1].shape[1]):
                isMark[indexLists[-1][-1][i,j,0], indexLists[-1][-1][i,j,1]] = False
        markImg = img.copy()
        markImg[isMark] = 0
        Image.fromarray(np.uint8(markImg)).save(file_path+"/"+str(id)+"_final2.png")

        # 生成gif图
        hIndex = len(imgLists)-1
        wIndex = len(imgLists[0])-1
        gifList = list()
        while chosenLists[hIndex][wIndex]!=-1:
            # 反向索引得到顺序删除的seam
            gifList.append(np.uint8(imgLists[hIndex][wIndex]))
            if chosenLists[hIndex][wIndex]==0:
                markImg = np.transpose(self.__markSeam(np.transpose(imgLists[hIndex-1][wIndex], (1,0,2)), seamLists[hIndex][wIndex]), (1,0,2))
                hIndex -= 1
            else:
                markImg = self.__markSeam(imgLists[hIndex][wIndex-1], seamLists[hIndex][wIndex])
                wIndex -= 1
            gifList.append(np.uint8(markImg))
        gifList.append(np.uint8(imgLists[0][0]))
        with imageio.get_writer(file_path+"/"+str(id)+"_output.gif", mode='I') as writer:
            for i in range(len(gifList)-1,-1,-1):
                writer.append_data(gifList[i]) #将图片写入writer，生成gif

def main():
    imgDict = dict()
    id = 13
    while id<MAX_NUM:
        img = Image.open('imgs/{}.png'.format(id))
        imgDict[id] = dict()
        imgDict[id]['img'] = np.asarray(img).copy().astype(int)
        img = Image.open('gt/{}.png'.format(id))
        imgArray = np.asarray(img).copy().astype(float)
        imgArray[imgArray>0] = 1
        imgDict[id]['mask'] = np.asarray(imgArray[:,:,0])
        id += 100

    model = SeamCarving()
    for id in imgDict.keys():
        height, width, _ = imgDict[id]['img'].shape
        areaProportion = np.sum(imgDict[id]['mask'])/imgDict[id]['mask'].size
        areaProportion += (1-areaProportion)/2
        height *= np.sqrt(areaProportion)
        width *= np.sqrt(areaProportion)
        print("#{} begins.".format(id))
        model.changeShape(imgDict[id]['img'], (int(height), int(width)), imgDict[id]['mask'], 'process', id)

if __name__=='__main__':
    main()

    