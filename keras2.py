from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler

(x_train,y_train),(x_test,y_test)=keras.datasets.fashion_mnist.load_data()
x_train=x_train[:50000]
x_test=x_test[:5000]
y_train=y_train[:50000]
y_test=y_test[:5000]
print(x_train.shape,y_train.shape)

x_train_transform=x_train/255
y_train_transform=y_train/255
x_test_transform=x_test/255
"""
x_train_transform=x_train.reshape((50000,-1))
x_test_transform=x_test.reshape((5000,-1))
scaler=StandardScaler()
x_train_transform=scaler.fit_transform(x_train_transform)
x_test_transform=scaler.transform(x_test_transform)
print(x_train_transform.shape,x_test_transform.shape)
x_train_transform=x_train_transform.reshape((50000,28,28))
x_test_transform=x_test_transform.reshape((5000,28,28))
"""
print(x_train_transform.shape,x_test_transform.shape)

for i in range(10):
    image=x_train[i,:,:]
    plt.imshow(image,cmap=plt.cm.gray)
    plt.show()
    print(y_train[i])

class_names=["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
 "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

model1=keras.models.Sequential()
model1.add(keras.layers.Flatten(input_shape=[28,28]))
model1.add(keras.layers.Dense(units=300,kernel_initializer='uniform',activation='relu'))
model1.add(keras.layers.Dense(units=100,kernel_initializer='uniform',activation='relu'))

model1.add(keras.layers.Dense(units=10,kernel_initializer='uniform',activation='softmax'))

print(model1.summary())
print(model1.layers[1].get_weights())

model1.compile(loss='sparse_categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])
history=model1.fit(x_train_transform,y_train,epochs=10,batch_size=10,validation_data=(x_test_transform,y_test))

df1=pd.DataFrame(data=history.history)
df1.plot(figsize=(8,5))
plt.show()

#import pydot,graphviz
#keras.utils.plot_model(model1)

model1.evaluate(x_test_transform,y_test)

y_pred=model1.predict_classes(x_test_transform[:10])
for i in range(10):
    image=x_test[i]
    plt.imshow(image,cmap=plt.cm.gray)
    plt.show()
    print('label :{0}\npredicted label:{1}'.format(y_test[i],class_names[y_pred[i]]))
    
