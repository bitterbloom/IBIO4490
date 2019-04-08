#!/home/afromero/anaconda3/bin/python

# read kaggle facial expression recognition challenge dataset (fer2013.csv)
# https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import ipdb
import warnings
warnings.filterwarnings("ignore")

def sigmoid(x):
    return 1/(1+np.exp(-x))

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
        emotion = 1 if int(emotion)==3 else 0 # Only for happiness
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

    x_val = x_train[26000:]
    x_train = x_train[0:26000]
    y_val = y_train[26000:]
    y_train = y_train[0:26000]
    

    print(x_train.shape[0], 'train samples')
    print(x_val.shape[0], 'validation samples')
    print(x_test.shape[0], 'test samples')

    # plt.hist(y_train, max(y_train)+1); plt.show()

    return x_train, y_train, x_val, y_val, x_test, y_test

class Model():
    def __init__(self):
        params = 48*48 # image reshape
        out = 1 # smile label
        self.lr = 0.01 # Change if you want
        np.random.seed(1)
        self.W = np.random.randn(params, out)
        self.b = np.random.randn(out)

    def forward(self, image):
        image = image.reshape(image.shape[0], -1)
        out = np.dot(image, self.W) + self.b
        return out

    def compute_loss(self, pred, gt):
        J = (-1/pred.shape[0]) * np.sum(np.multiply(gt, np.log(sigmoid(pred))) + np.multiply((1-gt), np.log(1 - sigmoid(pred))))
        return J

    def compute_gradient(self, image, pred, gt):
        image = image.reshape(image.shape[0], -1)
        W_grad = np.dot(image.T, pred-gt)/image.shape[0]
        self.W -= W_grad*self.lr

        b_grad = np.sum(pred-gt)/image.shape[0]
        self.b -= b_grad*self.lr

def train(model):
    x_train, y_train, x_val, y_val, x_test, y_test = get_data()
    batch_size = 100 # Change if you want
    epochs = 5000 # Change if you want
    avg_loss_train = []
    avg_loss_test = []
    for i in range(epochs):
        loss = []
        if (i%500) == 0:
            model.lr = model.lr/10
        for j in range(0,x_train.shape[0], batch_size):
            _x_train = x_train[j:j+batch_size]
            _y_train = y_train[j:j+batch_size]
            out = model.forward(_x_train)
            loss.append(model.compute_loss(out, _y_train))
            model.compute_gradient(_x_train, out, _y_train)

        out_val = model.forward(x_val)                
        loss_val = model.compute_loss(out_val, y_val)
        out_test = model.forward(x_test)                
        loss_test = model.compute_loss(out_test, y_test)

        print('Epoch {:6d}: {:.5f} | val: {:.5f}'.format(i, np.array(loss).mean(), loss_val))

        avg_loss_train.append(np.array(loss).mean())
        avg_loss_test.append(np.array(loss_test).mean())

    avg_loss_train = np.array(avg_loss_train)
    avg_loss_test = np.array(avg_loss_test)

    plot(avg_loss_train,avg_loss_test,epochs)

    import pickle
    params = {
        "W":model.W,
        "b":model.b
    }
    with open("params_bin.pickle","wb") as f:
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
    url = 'https://www.dropbox.com/s/7ftv9r3qoj39n5z/params_bin.pickle?dl=1'
    r = requests.get(url)
    with open("params_bin.pickle", "wb") as code:
            code.write(r.content)
    import pickle
    
    params = pickle.load(open( "params_bin.pickle", "rb" ))
    W = params["W"]
    b = params["b"]

    reshaped_data = x_test.reshape(x_test.shape[0], -1)
    out = np.dot(reshaped_data, W) + b
    predictions = sigmoid(out)

    y_true = np.array(y_test,dtype=np.int)

    from sklearn.metrics import precision_recall_curve
    P, R, _ = precision_recall_curve(y_true,predictions)

    from sklearn.metrics import f1_score
    F_measures = (2 * np.multiply(P,R)) / (np.add(P,R))
    F_measures = np.array(F_measures)
    F1 = np.nanmax(F_measures,axis=0)
    print(f'The F-measure was {F1:.4f}')

    pr = plt.figure()
    plt.plot(R,P)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve with F-measure = {F1:.4f}')
    pr.savefig('PRCurve.png', bbox_inches='tight')
    pr.savefig('PRCurve.pdf', bbox_inches='tight')
    plt.show()

    thresh = 0.5
    predictions[predictions >= thresh] = 1
    predictions[predictions < thresh] = 0

    plot_confusion_matrix(y_true, predictions, normalize=True, 
            classes=['1','0'],title='Normalized confusion matrix')

def demo():
    
    import os
    from skimage import color, io
    from skimage.transform import resize
    cwd = os.getcwd()

    x_demo = []
    y_demo = []

    for file in os.listdir('./images'):
        im = color.rgb2gray(io.imread(cwd + "/images/" + file))
        im = resize(im,(48,48))
        emotion = file.split("_")[0]
        if emotion == "happy":
            label = 1
        else:
            label = 0

        x_demo.append(im)
        y_demo.append(label)
    
    x_demo = np.array(x_demo, 'float64')
    y_demo = np.array(y_demo, 'float64')

    x_demo /= 255

    x_demo = x_demo.reshape(x_demo.shape[0], 48, 48)
    y_demo = y_demo.reshape(y_demo.shape[0], 1)
    
    print(x_demo.shape[0], 'wild samples')

    import requests
    url = 'https://www.dropbox.com/s/7ftv9r3qoj39n5z/params_bin.pickle?dl=1'
    r = requests.get(url)
    with open("params_bin.pickle", "wb") as code:
            code.write(r.content)
    import pickle
    
    params = pickle.load(open( "params_bin.pickle", "rb" ))
    W = params["W"]
    b = params["b"]

    reshaped_data = x_demo.reshape(x_demo.shape[0], -1)
    out = np.dot(reshaped_data, W) + b
    predictions = sigmoid(out)
    y_true = np.array(y_demo,dtype=np.int)

    from sklearn.metrics import precision_recall_curve
    P, R, _ = precision_recall_curve(y_true,predictions)

    from sklearn.metrics import f1_score
    F_measures = (2 * np.multiply(P,R)) / (np.add(P,R))
    F_measures = np.array(F_measures)
    F1 = np.nanmax(F_measures,axis=0)
    print(f'The F-measure was {F1}')

    pr = plt.figure()
    plt.plot(R,P)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve with F-measure = {F1:.4f}')
    plt.show()
    pr.savefig('PRCurve.png', bbox_inches='tight')
    pr.savefig('PRCurve.pdf', bbox_inches='tight')
    

    thresh = 0.5
    predictions[predictions >= thresh] = 1
    predictions[predictions < thresh] = 0

    plot_confusion_matrix(y_true, predictions, normalize=True, 
            classes=['0','1'],title='Normalized confusion matrix')

    
    
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
    import argparse
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--test', action ='store_true')
    parser.add_argument('--demo', action ='store_true')
    
    opts = parser.parse_args()
    
    if opts.test:
        test()
    elif opts.demo:
        demo()
    else:
        model = Model()
        train(model)
        test()
