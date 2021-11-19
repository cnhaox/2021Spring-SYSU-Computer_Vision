import numpy as np
import random
from PIL import Image
from segmentation import Segmentation
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression

MAX_NUM = 1000

class ForePredict():
    def __init__(self):
        self.__pcaModel = None
        self.__kmeansModel = None
        self.__predictModel = None
    
    def __transform(self, imgList, maskList):
        '''
        预处理数据，对图像进行分割、获得区域的向量表示和对应的标签
        '''
        model = Segmentation(k=10)
        data_x = list()
        data_y = list()
        print("data transform: ")
        for imgIndex in range(len(imgList)):
            print("#{} begins.".format(imgIndex))
            # 获取图片的全局直方图
            totalHint = np.zeros(512)
            tempImg = imgList[imgIndex].copy()
            tempIndex = np.floor(tempImg/32).astype(int)
            for i in range(512):
                rIndex = int(np.floor(i/64))
                gIndex = int(np.floor((i-rIndex*64)/8))
                bIndex = i-rIndex*64-gIndex*8
                totalHint[i] = np.sum(np.all(tempIndex==np.array([rIndex,gIndex,bIndex]),axis=2))
            # 归一化
            totalHint /= imgList[imgIndex].shape[0]*imgList[imgIndex].shape[1]
            # 获取分割区域，需要自动调整k值
            model.k = 150
            clusterIndexMap, clusterIdList = model.getCluster(imgList[imgIndex])
            while len(clusterIdList)<50 or len(clusterIdList)>70:
                if len(clusterIdList)<50:
                    model.k = int(model.k*0.8)
                    if model.k<=0:
                        break
                else:
                    model.k = int(model.k*1.2)
                    if model.k>=1000:
                        break
                clusterIndexMap, clusterIdList = model.getCluster(imgList[imgIndex])
            # 获取每个区域的标签(0: 背景；1: 前景)
            newMask, yList = model.getMaskedImg(imgList[imgIndex], maskList[imgIndex], clusterIndexMap, clusterIdList)
            for cId in clusterIdList:
                # 获取区域的局部直方图
                localHint = np.zeros((512))
                localIndex = tempIndex[clusterIndexMap==cId]
                for i in range(512):
                    rIndex = int(np.floor(i/64))
                    gIndex = int(np.floor((i-rIndex*64)/8))
                    bIndex = i-rIndex*64-gIndex*8
                    localHint[i] = np.sum(np.all(localIndex==np.array([rIndex, gIndex,bIndex]), axis=1))
                # 归一化
                localHint /= localIndex.shape[0]
                # 区域的向量表示
                data_x.append(np.append(totalHint,localHint))
            # 区域标签
            data_y += yList
        return np.array(data_x), np.array(data_y)
    
    def train(self, imgList, maskList, new_features = 20, bag_size=50):
        '''
        训练函数
        '''
        # 处理数据
        train_x, train_y = self.__transform(imgList, maskList)
        # pca降维
        self.__pcaModel = PCA(n_components=new_features).fit(train_x)
        train_x = self.__pcaModel.transform(train_x)
        # 构建visual bag
        self.__kmeansModel = KMeans(n_clusters=bag_size, random_state=0).fit(train_x)
        # 计算相似度，拼接得到新训练数据
        similarity = np.dot(train_x, self.__kmeansModel.cluster_centers_.T)
        similarity /= np.sum(self.__kmeansModel.cluster_centers_**2, axis=1)
        similarity /= np.sum(train_x**2, axis=1).reshape((-1,1))
        train_x = np.concatenate((train_x, similarity),axis=1)
        # 标准化
        self.__mean = np.mean(train_x,axis=0)
        self.__std = np.std(train_x, axis=0)
        train_x = (train_x-self.__mean)/self.__std
        # 使用逻辑回归
        self.__predictModel = LogisticRegression(random_state=0).fit(train_x, train_y)
        # 计算准确率和召回率
        pred_y = self.__predictModel.predict(train_x)
        acc = np.sum(pred_y==train_y)/train_y.size
        recall = np.sum(pred_y & train_y)/np.sum(pred_y)
        return acc, recall

    def test(self, imgList, maskList):
        '''
        测试函数
        '''
        test_x, test_y = self.__transform(imgList, maskList)
        # pca降维
        test_x = self.__pcaModel.transform(test_x)
        # 计算相似度，拼接数据
        similarity = np.dot(test_x, self.__kmeansModel.cluster_centers_.T)
        similarity /= np.sum(self.__kmeansModel.cluster_centers_**2, axis=1)
        similarity /= np.sum(test_x**2, axis=1).reshape((-1,1))
        test_x = np.concatenate((test_x, similarity), axis=1)
        test_x = (test_x-self.__mean)/self.__std
        # 计算准确率和召回率
        pred_y = self.__predictModel.predict(test_x)
        acc = np.sum(pred_y==test_y)/test_y.size
        recall = np.sum(pred_y & test_y)/np.sum(pred_y)
        return acc, recall

def main():
    random.seed(0)
    trainImgList = list()
    trainMaskList = list()
    testImgList = list()
    testMaskList = list()
    usedIdList = list()
    id = 13
    while id<MAX_NUM:
        usedIdList.append(id)
        id += 100
    for id in usedIdList:
        img = Image.open('imgs/{}.png'.format(id))
        testImgList.append(np.asarray(img))
        img = Image.open('gt/{}.png'.format(id))
        testMaskList.append(np.asarray(img))
    for i in range(200):
        id = random.randint(1, MAX_NUM)
        while id in usedIdList:
            id = random.randint(1,MAX_NUM)
        usedIdList.append(id)
        img = Image.open('imgs/{}.png'.format(id))
        trainImgList.append(np.asarray(img))
        img = Image.open('gt/{}.png'.format(id))
        trainMaskList.append(np.asarray(img))
    
    model = ForePredict()
    acc, recall = model.train(trainImgList, trainMaskList)
    print("train: acc={}; recall={}".format(acc,recall))
    acc, recall = model.test(testImgList, testMaskList)
    print("test: acc={}; recall={}".format(acc,recall))

if __name__=='__main__':
    main()