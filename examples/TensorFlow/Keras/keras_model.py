import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers

# ## Loading MNIST
# In[ ]:

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train / 255
x_test = x_test / 255


# In[3]:


print(x_train.shape)
print(y_train.shape)


# In[4]:


plt.imshow(x_train[0], cmap="Greys")


# ## Training with one-hot labels

# In[5]:


model_lr = tf.keras.models.Sequential(
    [
        layers.Input(x_train.shape[1:]),
        layers.Flatten(),
        layers.Dense(10, activation="softmax"),
    ]
)
model_lr.compile(
    optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
)
model_lr.summary()


# In[6]:


y_onehot_train = tf.one_hot(y_train, 10)
model_lr.fit(x_train, y_onehot_train)


# ## Training with sparse labels

# In[7]:


model_lr = tf.keras.models.Sequential(
    [
        layers.Input(x_train.shape[1:]),
        layers.Flatten(),
        layers.Dense(10, activation="softmax"),
    ]
)
model_lr.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)
model_lr.summary()


# In[8]:


# model_lr.fit(x_train, y_train)


# In[9]:


history_lr = model_lr.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=128,
    validation_data=(x_test, y_test),
    verbose=False,
)


# ## Review Traning Results

# In[11]:


plt.plot(history_lr.history["loss"], label="train")
plt.plot(history_lr.history["val_loss"], label="val")
plt.ylabel("loss")
plt.legend()
plt.show()

plt.plot(history_lr.history["accuracy"], label="train")
plt.plot(history_lr.history["val_accuracy"], label="val")
plt.ylabel("accuracy")
plt.legend()
plt.show()


# In[12]:


model_lr.evaluate(x_test, y_test)


# In[16]:


probs = model_lr.predict(x_test[:5])
preds = np.argmax(probs, axis=1)
for i in range(5):
    print(probs[i], " => ", preds[i])
    plt.imshow(x_test[i], cmap="Greys")
    plt.show()


# In[18]:


model_lr.predict(x_test[18].reshape(1, 28, 28))


# In[19]:


model_lr.predict(x_test[18:19])


# ## Adding Model Complexity

# In[20]:


model_mlp = tf.keras.models.Sequential(
    [
        layers.Input(x_train.shape[1:]),
        layers.Flatten(),
        layers.Dense(64, activation="elu"),
        layers.Dense(64, activation="elu"),
        layers.Dense(10, activation="softmax"),
    ]
)
model_mlp.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)
model_mlp.summary()


# In[21]:


history_mlp = model_mlp.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=128,
    validation_data=(x_test, y_test),
    verbose=False,
)


# In[22]:


plt.plot(history_mlp.history["loss"], label="train")
plt.plot(history_mlp.history["val_loss"], label="val")
plt.ylabel("loss")
plt.legend()
plt.show()

plt.plot(history_mlp.history["accuracy"], label="train")
plt.plot(history_mlp.history["val_accuracy"], label="val")
plt.ylabel("accuracy")
plt.legend()
plt.show()
