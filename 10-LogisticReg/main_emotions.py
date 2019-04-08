#!/home/afromero/anaconda3/bin/python

# read kaggle facial expression recognition challenge dataset (fer2013.csv)
# https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
import matplotlib as mpl
import ipdb
import warnings
warnings.filterwarnings("ignore")

def sigmoid(x):  
    return 1/(1+np.exp(-x))

def sigmoid_der(x):  
    return sigmoid(x) *(1-sigmoid (x))

def softmax(A):  
    expA = np.exp(A)
    return expA / expA.sum(axis=1, keepdims=True)

def get_data():
    # angry, disgust, fear, happy, sad, surprise, neutral

    import os
    if not os.path.isfile("./fer2013.csv"):
        import requests
        url = url = 'http://bcv001.uniandes.edu.co/fer2013.zip'
        r = requests.get(url)
        with open("fer2013.zip", "wb") as code:
                code.write(r.content)

        import zipfile
        zip_ref = zipfile.ZipFile("./fer2013.zip", 'r')
        zip_ref.extractall("./")
        zip_ref.close()
        os.remove("./fer2013.zip")

    with open("fer2013.csv") as f:
        content = f.readlines()

    lines = np.array(content)
    num_of_instances = lines.size
    print("number of instances: ",num_of_instances)
    print("instance length: ",len(lines[1].split(",")[1].split(" ")))

    x_train, y_train, x_test, y_test = [], [], [], []

    for i in range(1,num_of_instances):
        emotion, img, usage = lines[i].split(",")
        pixels = np.array(img.split(" "), 'float32')
        if 'Training' in usage:
            y_train.append(emotion)
            x_train.append(pixels)
        elif 'PublicTest' in usage:
            y_test.append(emotion)
            x_test.append(pixels)

    #------------------------------
    #data transformation for train and test sets
    x_train = np.array(x_train, 'float64')
    y_train = np.array(y_train, 'float64')
    x_test = np.array(x_test, 'float64')
    y_test = np.array(y_test, 'float64')

    x_train /= 255 #normalize inputs between [0, 1]
    x_test /= 255

    x_train = x_train.reshape(x_train.shape[0], 48, 48)
    x_test = x_test.reshape(x_test.shape[0], 48, 48)
    y_train = y_train.reshape(y_train.shape[0], 1)
    y_test = y_test.reshape(y_test.shape[0], 1)

    onhot_train = np.zeros((y_train.shape[0],7),dtype = int)
    onhot_test = np.zeros((y_test.shape[0],7),dtype = int)

    for i in range(y_train.shape[0]):
        onhot_train[i,int(y_train[i])] = 1
    
    for i in range(y_test.shape[0]):
        onhot_test[i,int(y_test[i])] = 1

    x_val = x_train[26000:]
    x_train = x_train[0:26000]
    lab_val = onhot_train[26000:,:]
    lab_train = onhot_train[0:26000,:]
    lab_test = onhot_test.copy()

    print(x_train.shape[0], 'train samples')
    print(x_val.shape[0], 'validation samples')
    print(x_test.shape[0], 'test samples')

    # plt.hist(y_train, max(y_train)+1); plt.show()

    return x_train, lab_train, x_val, lab_val, x_test, lab_test


    
class Model():
    def __init__(self):
        self.lr = 0.00001 # Change if you want
        np.random.seed(1)
  
        attributes = 48*48 
        hidden_nodes = 1  
        output_labels = 7
        self.wh = np.random.rand(attributes,hidden_nodes)  #W for the only hidden layer
        self.bh = np.random.randn(hidden_nodes) #b for the only hidden layer

        self.wo = np.random.rand(hidden_nodes,output_labels)  #W for the output
        self.bo = np.random.randn(output_labels) #b for the output

    def calc_loss(self,data,labels):
        data = data.reshape(data.shape[0], -1)
        zh = np.dot(data, self.wh) + self.bh #dot product of input vector and W hidden plus the bias hidden to obtain a hidden label
        ah = sigmoid(zh) #sigmoid the hidden label to obtain a number between 0 and 1

        zo = np.dot(ah, self.wo) + self.bo #dot product of the sigmoid to the hidden label and W output plus the bias output to obtain a number for each class
        ao = softmax(zo) #Normalize zo to a probability distribution between the classes that sums 1
        return np.sum(-labels * np.log(ao))

def train(model):
    x_train, y_train, x_val, y_val, x_test, y_test = get_data()
    batch_size = 100 # Change if you want
    epochs = 40000 # Change if you want
    avg_loss_train = []
    avg_loss_test = []
    for i in range(epochs):
        loss = []
        for j in range(0,x_train.shape[0], batch_size):
            _x_train = x_train[j:j+batch_size]
            _x_train = _x_train.reshape(_x_train.shape[0], -1)
            _y_train = y_train[j:j+batch_size]
            wh = model.wh
            bh = model.bh
            wo = model.wo
            bo = model.bo

            ############# feedforward

            # Phase 1
            zh = np.dot(_x_train, wh) + bh
            ah = sigmoid(zh)

            # Phase 2
            zo = np.dot(ah, wo) + bo
            ao = softmax(zo)

            ########## Back Propagation

            ########## Phase 1

            dcost_dzo = ao - _y_train #Derivate of the cost in terms of zo which is equal to the output of the softmax minus the onhot label
            dzo_dwo = ah #Derivate of zo in terms of wo which is equal to the sigmoid output of the hidden layer

            dcost_wo = np.dot(dzo_dwo.T, dcost_dzo) #Derivate of the cost in terms of wo which is a rule chain of the previous terms

            dcost_bo = dcost_dzo #Derivate of the cost in terms of bo which is the same as the derivate of the cost in terms of zo

            ########## Phases 2

            dzo_dah = wo #Derivate of zo in terms of ah is the same as wo
            dcost_dah = np.dot(dcost_dzo , dzo_dah.T) #Derivate of the cost in terms of ah is the chain rule with dcost_dzo and dzo_dah
            dah_dzh = sigmoid_der(zh) #Sigmoid derivative of zh is the same as the derivative of ah in terms of of zh as seen in Phase 1
            dzh_dwh = _x_train  #Derivate of zh in terms of wh is the train batch
            dcost_wh = np.dot(dzh_dwh.T, dah_dzh * dcost_dah) #Derivate of the cost in terms of wh is the chain rule with dzh_dwh and dcost_dzh that is equal to chain rule of dah_dzh and dcost_dah 

            dcost_bh = dcost_dah * dah_dzh #Derivate of cost in terms of bh is the chain rule of dcost_dah and dah_dzh which leaves dcost_dzh which will kill all terms in zh but the bh

            # Update Weights ================
            lr = model.lr
            model.wh -= lr * dcost_wh #update wh
            model.bh -= lr * dcost_bh.sum(axis=0) #update bh

            model.wo -= lr * dcost_wo #update wo
            model.bo -= lr * dcost_bo.sum(axis=0) #update bo
            
            loss.append(model.calc_loss(_x_train,_y_train))
              
        loss_val = model.calc_loss(x_val, y_val)             
        loss_test = model.calc_loss(x_test, y_test)   

        print('Epoch {:6d}: {:.5f} | val: {:.5f}'.format(i, np.array(loss).mean(), loss_val))

        avg_loss_train.append(np.array(loss).mean())
        avg_loss_test.append(np.array(loss_test).mean())

    avg_loss_train = np.array(avg_loss_train)
    avg_loss_test = np.array(avg_loss_test)

    plot(avg_loss_train,avg_loss_test,epochs)

    ipdb.set_trace()
    import pickle
    params = {
        "W":model.wo*model.wh,
        "b":model.bo*model.bh
    }
    with open("params_mult.pickle","wb") as f:
        pickle.dump( params, f)



def plot(loss_train,loss_test,epochs): # Add arguments
    # CODE HERE
    # Save a pdf figure with train and test losses

    los = plt.figure()
    plt.subplot(121)
    plt.scatter(range(0,epochs),loss_train)
    plt.title('Loss vs Epochs in Train')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.subplot(122)
    plt.scatter(range(0,epochs),loss_test)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss vs Epochs in Test')
    plt.subplots_adjust(wspace=0.4)
    plt.show()
    los.savefig('LossTestTrain.png', bbox_inches='tight')
    los.savefig('LossTestTrain.pdf', bbox_inches='tight')
    


def test():
    _, _, _, _, x_test, y_test = get_data()
    
    import requests
    url = 'https://www.dropbox.com/s/s9gtkmg7nrr98os/params_mult.pickle?dl=1'
    r = requests.get(url)
    with open("params_mult.pickle", "wb") as code:
            code.write(r.content)
    import pickle
    
    params = pickle.load(open( "params_mult.pickle", "rb" ))
    W = params["W"]
    b = params["b"]

    reshaped_data = x_test.reshape(x_test.shape[0], -1)
    out = np.dot(reshaped_data, W) + b
    predictions = softmax(out)
    pred_lab = np.nanmax(predictions,axis=1)
    pred_final = np.zeros(x_test.shape[0],dtype=int)
    for i in range(x_test.shape[0]):
        pred_final[i] = np.where(predictions[i,:]==pred_lab[i])[0][0]
    
    y_true = np.array(y_test,dtype=np.int)
    lab_final = np.zeros(x_test.shape[0],dtype=int)
    for i in range(x_test.shape[0]):
        lab_final[i] = np.where(y_true[i] == 1)[0]

    plot_confusion_matrix(lab_final, pred_final, normalize=True, 
            classes=['0','1','2','3','4','5','6'],title='Normalized confusion matrix')

def demo():
    
    import os
    from skimage import color, io
    from skimage.transform import resize
    cwd = os.getcwd()

    x_demo = []
    y_demo = []
    # angry, disgust, fear, happy, sad, surprise, neutral
    emotions={
        "angry":0,
        "disgust":1,
        "fear":2,
        "happy":3,
        "sad":4,
        "surprise":5,
        "neutral":6
    }
    for file in os.listdir('./images'):
        im = color.rgb2gray(io.imread(cwd + "/images/" + file))
        im = resize(im,(48,48))
        emotion = file.split("_")[0]
        label = emotions[emotion]

        x_demo.append(im)
        y_demo.append(label)
    
    x_demo = np.array(x_demo, 'float64')
    y_demo = np.array(y_demo, 'float64')

    x_demo /= 255

    x_demo = x_demo.reshape(x_demo.shape[0], 48, 48)
    y_demo = y_demo.reshape(y_demo.shape[0], 1)
    
    print(x_demo.shape[0], 'wild samples')

    import requests
    url = 'https://www.dropbox.com/s/s9gtkmg7nrr98os/params_mult.pickle?dl=1'
    r = requests.get(url)
    with open("params_mult.pickle", "wb") as code:
            code.write(r.content)
    import pickle
    params = pickle.load(open( "params_mult.pickle", "rb" ))
    W = params["W"]
    b = params["b"]

    reshaped_data = x_demo.reshape(x_demo.shape[0], -1)
    out = np.dot(reshaped_data, W) + b
    predictions = softmax(out)
    pred_lab = np.nanmax(predictions,axis=1)
    pred_final = np.zeros(x_demo.shape[0],dtype=int)
    for i in range(x_demo.shape[0]):
        pred_final[i] = np.where(predictions[i,:]==pred_lab[i])[0][0]
    
    y_true = np.array(y_demo,dtype=np.int)
    plot_confusion_matrix(y_true, pred_final, normalize=True, 
            classes=['0','1','2','3','4','5','6'],title='Normalized confusion matrix')

    
    
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'
    # Compute confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
        ACA = cm.diagonal().mean()
        print(f"ACA = {ACA:.4f}")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=(f"" + title + " ACA = " + f"{ACA:.4f}"),
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.show()
    fig.savefig('ConfMat.png', bbox_inches='tight')
    fig.savefig('ConfMat.pdf', bbox_inches='tight')
    return ax
    


if __name__ == '__main__':
    mpl.use('Agg')
    model = Model()
    train(model)
    test()

