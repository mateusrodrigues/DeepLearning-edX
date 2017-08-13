# Import the relevant components
from __future__ import print_function # Use a function definition from future version (say 3.x from 2.7 interpreter)
from IPython.display import Image
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

import cntk as C

# Select the right target device
if 'TEST_DEVICE' in os.environ:
    if os.environ['TEST_DEVICE'] == 'cpu':
        C.device.try_set_default_device(C.device.cpu())
    else:
        C.device.try_set_default_device(C.device.gpu(0))

# Test for CNTK version
print('You are using CNTK {}'.format(C.__version__))
if not C.__version__ == '2.1':
    raise Exception('This lab is designed to work with 2.0. Current version: ' + C.__version__)

# Ensure we always get the same amount of randomness
np.random.seed(0)
C.cntk_py.set_fixed_random_seed(1)
C.cntk_py.force_deterministic_algorithms()

# Define the data dimensions
input_dim = 784
num_output_classes = 10

# Read a CTF formetted text (as mentioned above) using the CTF deserializer from a file
def create_reader(path, is_training, input_dim, num_label_classes):
    labelStream = C.io.StreamDef(field = 'labels', shape = num_label_classes, is_sparse = False)
    featureStream = C.io.StreamDef(field = 'features', shape = input_dim, is_sparse = False)

    deserializer = C.io.CTFDeserializer(path, C.io.StreamDefs(labels = labelStream, features = featureStream))

    return C.io.MinibatchSource(deserializer, randomize = is_training, max_sweeps = C.io.INFINITELY_REPEAT if is_training else 1)

# Ensure the training and test data is generated and available for this lab
# We search in two locations in the toolkit for the cached MNIST data set
data_found = False

for data_dir in [os.path.join('..', 'Examples', 'Image', 'DataSets', 'MNIST'),
                 os.path.join('data', 'MNIST')]:
    train_file = os.path.join(data_dir, 'Train-28x28_cntk_text.txt')
    test_file = os.path.join(data_dir, 'Test-28x28_cntk_text.txt')
    if os.path.isfile(train_file) and os.path.isfile(test_file):
        data_found = True
        break

if not data_found:
    raise Exception('Please generate the data by downloading and serializing the MNIST dataset')

print('Data directory is {0}'.format(data_dir))

# Create CNTK inputs
input = C.input_variable(input_dim)
label = C.input_variable(num_output_classes)

def create_model(features):
    with C.layers.default_options(init = C.glorot_uniform()):
        r = C.layers.Dense(num_output_classes, activation = None)(features)
        return r

# Scale the input to 0-1 range by dividing each pixel by 255
# z represents the output of the network -> z = Wx' + b
input_s = input/255
squared_input = C.square(input_s)
sqrted_input = C.sqrt(input_s)

normalized_input = C.splice(input_s, squared_input, sqrted_input)
z = create_model(normalized_input)

# Define loss to minimize the cross-entropy between the label and predicted
# probability by the network
loss = C.cross_entropy_with_softmax(z, label)

# Define the evaluation (metric) function to report how well our model is performing
label_error = C.classification_error(z, label)

# Instantiate the trainer object to drive the model training
learning_rate = 0.2
lr_schedule = C.learning_rate_schedule(learning_rate, C.UnitType.minibatch)
learner = C.sgd(z.parameters, lr_schedule)
trainer = C.Trainer(model = z, criterion = (loss, label_error), parameter_learners = [ learner ])

# Define a utility function to compute the moving average sum.
# A more efficient implementation is possible with np.cumsum() function
def moving_average(a, w=5):
    if len(a) < w:
        return a[:] # Need to send a copy of the array
    return [val if idx < w else sum(a[(idx-w):idx])/w for idx, val in enumerate(a)]

# Defines a utility that prints the training progress
def print_training_progress(trainer, mb, frequency, verbose=1):
    training_loss = 'NA'
    eval_error = 'NA'

    if mb%frequency == 0:
        training_loss = trainer.previous_minibatch_loss_average
        eval_error = trainer.previous_minibatch_evaluation_average
        if verbose:
            print('Minibatch: {0}, Loss: {1:.4f}, Error: {2:.2f}%'.format(mb, training_loss, eval_error*100))

    return mb, training_loss, eval_error

# Initialize the parameters for the trainer
minibatch_size = 64
num_samples_per_sweep = 60000
num_sweeps_to_train_with = 10
num_minibatches_to_train = (num_samples_per_sweep * num_sweeps_to_train_with) / minibatch_size

# Create the reader to training data set
reader_train = create_reader(train_file, True, input_dim, num_output_classes)

# Map the data streams to the input and labels
input_map = {
    label : reader_train.streams.labels,
    input : reader_train.streams.features
}

# Run the trainer on and perform model training
training_progress_output_freq = 500

plotdata = { 'batchsize':[], 'loss':[], 'error':[] }

for i in range(0, int(num_minibatches_to_train)):

    # Read a minibatch from the training data file
    data = reader_train.next_minibatch(minibatch_size, input_map = input_map)

    trainer.train_minibatch(data)
    batchsize, loss, error = print_training_progress(trainer, i, training_progress_output_freq, verbose = 1)

    if not (loss == 'NA' or error == 'NA'):
        plotdata['batchsize'].append(batchsize)
        plotdata['loss'].append(loss)
        plotdata['error'].append(error)

# Compute the moving average loss to smooth out the noise in SGD
plotdata['avgloss'] = moving_average(plotdata['loss'])
plotdata['avgerror'] = moving_average(plotdata['error'])

# Plot the training loss and the training error
plt.figure(1)
plt.subplot(211)
plt.plot(plotdata['batchsize'], plotdata['avgloss'], 'b--')
plt.xlabel('Minibatch number')
plt.ylabel('Loss')
plt.title('Minibatch run vs. Training loss')
# plt.show()

plt.subplot(212)
plt.plot(plotdata['batchsize'], plotdata['avgerror'], 'r--')
plt.xlabel('Minibatch number')
plt.ylabel('Label Prediction Error')
plt.title('Minibatch run vs. Label Prediction Error')
plt.show()

# Evaluation/Testing

# Read the training data
reader_test = create_reader(test_file, False, input_dim, num_output_classes)

test_input_map = {
    label : reader_test.streams.labels,
    input : reader_test.streams.features
}

# Test data for trained model
test_minibatch_size = 512
num_samples = 10000
num_minibatches_to_test = num_samples // test_minibatch_size
test_result = 0.0

for i in range(num_minibatches_to_test):

    # We are loading test data in batches specified by test_minibatch_size
    # Each data point in the minibatch is a MNIST digit image of 784 dimensions 
    # with one pixel per dimension that we will encode / decode with the 
    # trained model.
    data = reader_test.next_minibatch(test_minibatch_size, input_map = test_input_map)

    eval_error = trainer.test_minibatch(data)
    test_result = test_result + eval_error

# Average of evaluation errors of all test minibatches
print('Average test error: {0:.2f}%'.format(test_result*100 / num_minibatches_to_test))

# Route the network output through a softmax function to map the aggregated activations
# across the network to probabilities across the 10 classes
out = C.softmax(z)

# Evaluating a small minibatch from the test data to see how the model is performing

# Read the data for evaluation
reader_eval = create_reader(test_file, False, input_dim, num_output_classes)

eval_minibatch_size = 25
eval_input_map = { input: reader_eval.streams.features }

data = reader_test.next_minibatch(eval_minibatch_size, input_map = test_input_map)

img_label = data[label].asarray()
img_data = data[input].asarray()
predicted_label_prob = [ out.eval(img_data[i]) for i in range(len(img_data)) ]

# Find the index with the maximum value for both predicted as well as the ground truth
pred = [ np.argmax(predicted_label_prob[i]) for i in range(len(predicted_label_prob)) ]
gtlabel = [ np.argmax(img_label[i]) for i in range(len(img_label)) ]

print('Label    :', gtlabel[:25])
print('Predicted:', pred)

# Plot a random image
sample_number = 5
plt.imshow(img_data[sample_number].reshape(28,28), cmap='gray_r')
plt.axis('off')

img_gt, img_pred = gtlabel[sample_number], pred[sample_number]
print('Image Label:', img_pred)