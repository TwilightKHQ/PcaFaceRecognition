# coding=gbk
"""���ڣУã��㷨������ʶ��
@author:      jingyiliscut@gmail.com
@time:        2012.10
"""
from PIL import Image, ImageDraw
import numpy
import random
import math
import os
import sys

IMAGE_SIZE = (40, 40)


def createDatabase(path, number):
    """��ָ����·��path�д���number������˳������������ͼƬ,��Ϊһ������"""
    # path �������ݵ�λ��
    # number ÿ���˺���10��ͼƬ, ѵ����ȡǰn��
    imageMatrix = []
    for i in range(1, 40 + 1):
        for j in range(1, number + 1):
            image = Image.open(path + str(i) + "/" + str(j) + '.pgm')
            image = image.resize(IMAGE_SIZE)  # ��СͼƬ
            grayImage = image.convert('L')
            imageArray = list(grayImage.getdata())  # ת��Ϊһ��һά���飬����������
            imageMatrix.append(imageArray)

    imageMatrix = numpy.array(imageMatrix)  # ת��Ϊ��ά������Ϊͼ������ֵ�������У�û��Ϊһ��ͼ
    # print imageMatix

    return imageMatrix


def eigenfaceCore(Matrix):
    """ͨ���õ���ͼ�����ѵ��������"""
    trainNumber, perTotal = numpy.shape(Matrix)  # ����ͼ��ĸ�������ÿ��ͼ��Ĵ�С
    print(trainNumber)

    """�����м���ƽ������"""
    meanArray = Matrix.mean(0)  # 0�����м���ƽ����1�����м���ƽ��

    """����ÿ��������ƽ�������Ĳ�"""
    diffMatrix = Matrix - meanArray

    """����Э�������C�����L"""
    # diffMatrixTranspose = numpy.transpose(diffMatrix) #����ת��
    diffMatrix = numpy.mat(diffMatrix)  # �����������͵�����
    L = diffMatrix * diffMatrix.T  # ʹ�˵õľ����С
    eigenvalues, eigenvectors = numpy.linalg.eig(L)  # ��������v[:,i]��Ӧ����ֵw[i]

    """����õ�������ֵ��������������˳��
        ��һ����������ֵ����1����ȡ��������"""
    # ��Ϊ�������������ÿ����һ������������
    # ������Ҫת�ú󣬱�Ϊһ��list,Ȼ��ͨ��pop������
    # ɾ�����е�һ�У�����任ת��ȥ
    eigenvectors = list(eigenvectors.T)
    pop_position = []
    for i in range(0, trainNumber, -1):
        if eigenvalues[i] < 1:
            eigenvectors.pop(i)

    eigenvectors = numpy.array(eigenvectors)  # �����޷�ֱ�Ӵ���һά�ľ���������Ҫһ���������
    eigenvectors = numpy.mat(eigenvectors).T

    """������������,Ҳ���Ǽ����C
        ���ֱ任�����˼������"""
    # print numpy.shape(diffMatrix)
    # print numpy.shape(eigenvectors)
    eigenfaces = diffMatrix.T * eigenvectors
    return eigenfaces


def recognize(testIamge, Matrix, eigenface):
    """testIamge,Ϊ����ʶ��Ĳ���ͼƬ
    MatrixΪ����ͼƬ���ɵľ���
    eigenfaceΪ������
       ����ʶ������ļ����к�"""

    """�����м���ƽ������"""
    meanArray = Matrix.mean(0)  # 0�����м���ƽ����1�����м���ƽ��

    """����ÿ��������ƽ�������Ĳ�"""
    diffMatrix = Matrix - meanArray

    """ȷ���������˺��ͼƬ��Ŀ"""
    perTotal, trainNumber = numpy.shape(eigenface)
    print(trainNumber)

    """��ÿ������ͶӰ�������ռ�"""
    projectedImage = eigenface.T * diffMatrix.T

    # print numpy.shape(projectedImage)
    """Ԥ�������ͼƬ������ӳ�䵽�����ռ���"""
    testimage = Image.open(testIamge)
    testimage = testimage.resize(IMAGE_SIZE)
    grayTestImage = testimage.convert('L')
    testImageArray = list(grayTestImage.getdata())  # ת��Ϊһ��һά���飬����������
    testImageArray = numpy.array(testImageArray)

    differenceTestImage = testImageArray - meanArray
    # ת��Ϊ������ڽ������ĳ˷�����
    differenceTestImage = numpy.array(differenceTestImage)
    differenceTestImage = numpy.mat(differenceTestImage)

    projectedTestImage = eigenface.T * differenceTestImage.T
    # print numpy.shape(projectedImage)
    # print numpy.shape(projectedTestImage)

    """����ŷʽ���������ƥ�������"""
    distance = []
    for i in range(0, trainNumber):
        q = projectedImage[:, i]
        temp = numpy.linalg.norm(projectedTestImage - q)  # ���㷶��
        distance.append(temp)

    minDistance = min(distance)
    index = distance.index(minDistance)

    return index + 1  # ����index�Ǵ�0��ʼ��


if __name__ == "__main__":
    TrainNumber = 8
    trainingSetPath = "ORL_Faces/s"
    Matrix = createDatabase(trainingSetPath, TrainNumber)
    eigenface = eigenfaceCore(Matrix)
    dir_position = random.randint(1, 40)
    number_position = random.randint(9, 10)
    print("dir_position: " + str(dir_position) + " number_position: " + str(number_position))
    test_image = trainingSetPath + str(dir_position) + "/" + str(number_position) + '.pgm'
    index = recognize(test_image, Matrix, eigenface)
    print("dir: " + str(math.ceil(index / TrainNumber)) + " number: " + str(index % TrainNumber))
