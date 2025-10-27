from Utils import *

TestData = pd.read_csv('test.csv')
TestData = np.array(TestData)
TestData = TestData.T
m,n = TestData.shape

TestData_X = TestData[0:n]
TestData_X = TestData_X / 255

l1 = layers(784) # Input Layer
l2 = layers(128) # Hidden Layer
l3 = layers(64) # Hidden Layer
l4 = layers(10) # Output Layer

l2.weights(l1.layerSize)
l3.weights(l2.layerSize)
l4.weights(l3.layerSize)

l1.A = TestData_X

with open ('parameters.json','r') as file:
    Js_Object = js.load(file)

l2.W = np.array(Js_Object['W2'])
l2.b = np.array(Js_Object['B2'])

l3.W = np.array(Js_Object['W3'])
l3.b = np.array(Js_Object['B3'])

l4.W = np.array(Js_Object['W4'])
l4.b = np.array(Js_Object['B4'])

forwardPropogation(l1,l2,l3,l4,n)


Result = get_predictions(l4.A)


import matplotlib.pyplot as plt

i = 0
switch = True
while switch:
    currentImage = TestData_X[:, i].reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(currentImage)
    plt.title(f"{i + 1} Prediction: {Result[i]}")
    plt.show()

    switch = input("Do you want to continue: ")
    switch = False if switch.strip().lower() == "exit" else True
    i += 1
