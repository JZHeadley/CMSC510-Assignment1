import numpy as np


def normalize(dataset):
    for i in range(0, dataset.__len__()):
        for j in range(0, dataset[i].__len__()):
            if dataset[i][j] > 0:
                dataset[i][j] = 1
    return dataset


def flatten(arrayOfMatrix):
    flattened = []
    for i in range(0, arrayOfMatrix.__len__()):
        flat_x = []
        for j in range(0, arrayOfMatrix[i].__len__()):
            for k in range(0, arrayOfMatrix[i][j].__len__()):
                flat_x.append(arrayOfMatrix[i][j][k])
        flattened.append(flat_x)
    return flattened


def extractClass(x, y, toExtract):
    classSamples = []
    for i in range(0, x.__len__()):
        if y[i] == toExtract:
            classSamples.append(x[i])
    return classSamples


def computeAccuracy(trueY, predY):
    numElements = trueY.__len__()
    correctPredictions = 0
    for i in range(0, numElements):
        correctPredictions += (trueY[i] == predY[i])
    return (correctPredictions/numElements)


def extractMine(x, y, class1, class2):
    x_mine = []
    y_mine = []
    for i in range(0, x.__len__()):
        if y[i] == class1 or y[i] == class2:
            x_mine.append(x[i])
            if(y[i] == class1):
                y_mine.append(1)
            elif(y[i] == class2):
                y_mine.append(0)
    return (x_mine, y_mine)


def featureSelection(x):
    x_selected = []
    highestX = 0
    highestY = 0
    for i in range(0, x.__len__()):
        selected_val = np.array(x[i]).compress(
            ~np.all(x[i] == 0, axis=0), axis=1)
        selected_val = selected_val[~np.all(selected_val == 0, axis=1)]
        if selected_val.shape[0] > highestX:
            highestX = selected_val.shape[0]
        if selected_val.shape[1] > highestY:
            highestY = selected_val.shape[1]
        x_selected.append(selected_val)
    print("We have", highestX, "x elements and", highestY, "y elements")
    pad_selected = []
    for i in range(0, x_selected.__len__()):
        x_shape = x_selected[i].shape[0]
        y_shape = x_selected[i].shape[1]
        padded_val = np.pad(
            x_selected[i], ((0, highestX-x_shape), (0, highestY-y_shape)), 'constant')
        pad_selected.append(padded_val)

    return pad_selected
