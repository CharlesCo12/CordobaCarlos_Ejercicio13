import matplotlib.pyplot as plt
import sklearn.datasets as skdata
import numpy as np
import sklearn.metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm

# lee los numeros
numeros = skdata.load_digits()

# lee los labels
target = numeros['target']

# lee las imagenes
imagenes = numeros['images']

# cuenta el numero de imagenes total
n_imagenes = len(target)

# para poder correr PCA debemos "aplanar las imagenes"
data = imagenes.reshape((n_imagenes, -1)) # para volver a tener los datos como imagen basta hacer data.reshape((n_imagenes, 8, 8))

# Split en train/test
x_80, x_test_f, y_80, y_test_f = train_test_split(data, target, train_size=0.8)
x_train, x_test, y_train, y_test = train_test_split(x_80,y_80, train_size=0.5)

# Reescalado de los datos
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)
x_test_f = scaler.transform(x_test_f)

cov = np.cov(x_train.T)
valores, vectores = np.linalg.eig(cov)
valores = np.real(valores)
vectores = np.real(vectores)
ii = np.argsort(-valores)
valores = valores[ii]
vectores = vectores[:,ii]

pr_train=np.dot(x_train,vectores)
pr_test=np.dot(x_test,vectores)
pr_test_f=np.dot(x_test_f,vectores)

c=np.logspace(-2,1,100)
f1=[]
for i in c:
    clf=svm.SVC(C=i,kernel='linear')
    clf.fit(pr_train[:,:10],y_train.T)
    a=clf.predict(pr_test[:,:10])
    f1.append(sklearn.metrics.f1_score(y_test, a, average='macro'))
i=np.where(f1==np.max(f1))[0]
c_max=np.mean(c[i])
clf_f=svm.SVC(C=c_max,kernel='linear')
clf_f.fit(pr_train[:,:10],y_train.T)
y_predict_f=clf_f.predict(pr_test_f[:,:10])

sklearn.metrics.plot_confusion_matrix(clf_f,pr_test_f[:,:10],y_test_f,normalize='true',values_format='.2g')
plt.title('Matriz de Confusión para C={:.3f}'.format(c_max))
plt.show()
plt.savefig('loquequieran.png')