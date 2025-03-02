from tensorflow.keras.datasets import cifar100  
import matplotlib.pyplot as plt                 

# load CIFAR-100 
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

print("Training data shape:", x_train.shape)  # (50000, 32, 32, 3)
print("Test data shape:", x_test.shape)       # (10000, 32, 32, 3)

plt.imshow(x_train[0])
plt.title("Sample Image from CIFAR-100")
plt.axis('off')
plt.savefig("sample_image.png")
plt.show()