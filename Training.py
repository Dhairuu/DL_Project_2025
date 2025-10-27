from Utils import *

data = pd.read_csv('train.csv')
data = np.array(data)
data_train = data.T
m,n = data.shape
X = data_train[1:n]
Y = data_train[0]
X = X / 255


l1 = layers(784) # Input Layer
l2 = layers(128) # Hidden Layer
l3 = layers(64) # Hidden Layer
l4 = layers(10) # Output Layer

l2.weights(l1.layerSize)
l3.weights(l2.layerSize)
l4.weights(l3.layerSize)

l1.A = X

# L = 0.2
# decay_factor = 1
# decay_step = 200


# # L_base = 0.2           # initial learning rate
# decay_step = 200       # number of epochs for one sine cycle
# amplitude = 0.1        # how much the LR oscillates (adjustable)
# min_lr = 0.05


# L_max = 0.2     # initial / max learning rate
# L_min = 0.1     # minimum learning rate
# decay_step = 200 

L_max = 0.5      # highest learning rate (when accuracy is low)
L_min = 0.01     # smallest learning rate (when accuracy is near 100%)
L = L_max         # initialize
decay_smooth = 0.9  # optional smoothing factor to avoid sudden jumps

import math

for epoch in range(600):

    
    # if ipoch > 0 and ipoch % decay_step == 0:
    #     L = L * decay_factor
    #     print(f"--- Learning rate decayed to L = {L:.4f} ---")

    # phase = (epoch % decay_step) / decay_step  # normalized 0→1 for each cycle
    # L = min_lr + amplitude * (1 + math.sin(2 * math.pi * phase)) / 2

    # phase = (epoch % decay_step) / decay_step
    # L = L_min + 0.5 * (L_max - L_min) * (1 + math.cos(math.pi * phase))

    forwardPropogation(l1,l2,l3,l4,m)
    BackwardPropogation(l1,l2,l3,l4,Y,m)

    accuracy = get_accuracy(get_predictions(l4.A), Y)

    # Dynamic learning rate based on accuracy
    target_L = L_min + (L_max - L_min) * (1 - accuracy) ** 0.5

    print(f"{round(decay_smooth, 3)} * {round(L, 3)} + ({round(1 - decay_smooth, 3)}) * {round(target_L, 3)} =", end =" ")
    # Smooth out the transition to prevent oscillations
    L = decay_smooth * L + (1 - decay_smooth) * target_L

    print(f"{round(L, 3)}")

    gradientDescent(l1,l2,l3,l4,L)
    if epoch % 50 == 0:
        print(round(L,3))
        print(epoch)
        print(f"Accuracy: {accuracy}")

print("Accuracy: ", get_accuracy(get_predictions(l4.A), Y))


Js_Object = {}


with open('parameters.json','w') as file:
    Js_Object['W2'] = l2.W.tolist()
    Js_Object['B2'] = l2.b.tolist()
    Js_Object['W3'] = l3.W.tolist()
    Js_Object['B3'] = l3.b.tolist()
    Js_Object['W4'] = l4.W.tolist()
    Js_Object['B4'] = l4.b.tolist()
    js.dump(Js_Object,file)