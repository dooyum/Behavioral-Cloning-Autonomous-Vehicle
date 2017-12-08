import csv
import cv2
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split

steering_correction = 0.25
image_path_prefix = './car_simulator_data/IMG/'

def generator(samples, batch_size=32):
	num_samples = len(samples)
	while 1: # Loop forever so the generator never terminates
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset:offset+batch_size]
			images = []
			steering_angles = []
			for batch_sample in batch_samples:
				steering = float(batch_sample[3])
				# randomly choose camera
				# selects center camera 10% of the time
				if np.random.random() < 0.1:
					camera_index = 0
				else: # randomly select left or right camera
					if np.random.random() > 0.5:
						camera_index = 1
						steering += steering_correction
					else:
						camera_index = 2
						steering -= steering_correction

				image = cv2.imread(image_path_prefix + batch_sample[camera_index].split('/')[-1])
				#randomly flip image
				if np.random.random() > 0.5:
					image = np.fliplr(image)
					steering = -steering

				images.append(image)
				steering_angles.append(steering)

			X_features = np.array(images)
			y_labels = np.array(steering_angles)
			yield sklearn.utils.shuffle(X_features, y_labels)

lines = []
# The number of sample runs that were collected to train the data.
# each driving log from the data is included in the training folder 
# and named driving_log_<run number>.csv. All camera images can be 
# put into the same folder.
sample_runs = 9
for i in range(sample_runs):
	with open('./car_simulator_data/driving_log_'+str(i)+'.csv') as csvfile:
		reader = csv.reader(csvfile)
		for line in reader:
			lines.append(line)

train_samples, validation_samples = train_test_split(lines, test_size=0.3)
train_generator = generator(train_samples)
validation_generator = generator(validation_samples)

# Histogram
num_samples = len(train_samples)
samples_generated = 0

steering_angles = np.array([])

while samples_generated < num_samples:
    X_batch, y_batch = next(train_generator)
    if steering_angles.any():
       steering_angles = np.concatenate([steering_angles, y_batch])
    else:
       steering_angles = y_batch
    samples_generated += y_batch.shape[0]

plt.title("Steering Angles")
plt.hist(steering_angles)
plt.xlabel('Angle')
plt.ylabel('Count')
plt.plot()
pp = PdfPages('history.pdf')
plt.savefig(pp, format='pdf')
pp.close()

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Cropping2D(cropping=((60,20), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: (x / 255.0) - 0.5))
model.add(Convolution2D(24,3,3,subsample=(2,2),activation='relu'))
model.add(Dropout(0.25))
model.add(MaxPooling2D())
model.add(Convolution2D(36,3,3,subsample=(2,2),activation='relu'))
model.add(Convolution2D(48,3,3,subsample=(2,2),activation='relu'))
model.add(Convolution2D(64,2,2,activation='relu'))
model.add(Convolution2D(120,2,2,activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=8)

model.save('model.h5')
