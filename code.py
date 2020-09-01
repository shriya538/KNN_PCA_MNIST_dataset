import struct
import random
from scipy.spatial import distance
import sys
import numpy as np
from sklearn.decomposition import PCA


def calculate_vote(point,pca_train_images):
    distance={}
    for i in range(len(pca_train_images)):
        dist = 0
        for x in range(len(point)):
            dist += (point[x] - pca_train_images[i][x])**2
        distance[i] = 1/dist
    return distance




def read_and_flatten (path):

    with open(path+'/train-images-idx3-ubyte', 'rb') as f:
        buffer = f.read(8)
        magic, num_images = struct.unpack(">II", buffer)
        buffer = f.read(8)
        num_row,num_column=struct.unpack(">II", buffer)
        data=f.read()
        images = np.frombuffer(data, dtype=np.uint8).reshape(num_images, num_row, num_column)[:800]

    with open(path+'/train-labels-idx1-ubyte', 'rb') as f:
        bytes = f.read(8)
        magic, size = struct.unpack(">II", bytes)
        labeldata = f.read()
        labels = np.frombuffer(labeldata, dtype=np.uint8)[:800]

    return [image.flatten() for image in images],labels

def split_data_train_test(images, labels, n):

    test_images=images[:n]
    test_labels=labels[:n]
    train_images=images[n:]
    train_labels=labels[n:]
    return test_images,test_labels,train_images,train_labels


if __name__ == '__main__':

    K= int(sys.argv[1])
    D=int(sys.argv[2])
    N=int(sys.argv[3])
    path=sys.argv[4]

    images,labels=read_and_flatten(path)
    test_images,test_labels,train_images,train_labels=split_data_train_test(images,labels,N)
    sum=0
    for num in test_images[0]:
        sum+=num
    a=sum/784
    model=PCA(n_components=D,svd_solver='full').fit(train_images)
    pca_train_images=model.transform(train_images)
    pca_test_images=model.transform(test_images)
    predicted_labels=[]

    for point in pca_test_images:
        vote=calculate_vote(point,pca_train_images)
        vote=sorted(vote.items(),key=lambda x: x[1],reverse=True)[:K]
        new_vote=[]
        for x in vote:
            new_vote.append(train_labels[x[0]])
        max_label=-1
        max_freq=0
        for l in set(new_vote):
            if max_freq<new_vote.count(l):
                max_freq=new_vote.count(l)
                max_label=l
        predicted_labels.append(max_label)

    res = []
    for x, y in zip(predicted_labels, test_labels):
        res.append([x, y])

    output_file= open("results.txt", 'w')

    for element in res:
        output_file.write(str(element[0]))
        output_file.write(' ')
        output_file.write(str(element[1]))
        output_file.write('\n')


    output_file.close()
    



    

























