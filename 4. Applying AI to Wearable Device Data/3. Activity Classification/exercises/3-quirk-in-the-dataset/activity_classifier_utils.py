import os

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import scipy.signal
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut


def LoadWristPPGDataset():
  """Load the Wrist PPG Dataset.

  Found on Physionet at https://physionet.org/content/wrist/1.0.0/

  Returns:
    A list of datasets where each element is a session of accelerometer from
    one subject performing one activity. Each element is a 3-tuple of:
    subject: (string) The subject id
    activity: (string) The activity class being performed
    df: (pd.DataFrame) A pandas DataFrame with column headers ['accx', 'accy',
      'accz']
      containing accelerometer data sampled at 256 Hz.
  """
  # This relative path is brittle. Depending on where the IPython kernel's 
  # working directory is, this may be wrong. If you are working offline you
  # should change this to the absolute path of the dataset.
  data_dir = 'data'
  filenames = [os.path.splitext(f)[0] for f in sorted(os.listdir(data_dir))]
  data = []
  for f in filenames:
    subject = f.split('_')[0]
    activity = f.split('_')[1]
    path = os.path.join(data_dir, f + '.csv')
    df = pd.read_csv(path)
    df = df.loc[: df.last_valid_index()]
    data.append((subject, activity, df))
  return data


def LowpassFilter(signal, fs):
  b, a = sp.signal.butter(3, 12, btype='lowpass', fs=fs)
  return sp.signal.filtfilt(b, a, signal)


def Featurize(accx, accy, accz, fs):
  """Featurization of the accelerometer signal.

  Args:
      accx: (np.array) x-channel of the accelerometer.
      accy: (np.array) y-channel of the accelerometer.
      accz: (np.array) z-channel of the accelerometer.
      fs: (number) the sampling rate of the accelerometer

  Returns:
      n-tuple of accelerometer features
  """

  accx = LowpassFilter(accx, fs)
  accy = LowpassFilter(accy, fs)
  accz = LowpassFilter(accz, fs)

  # The mean of each channel
  mn_x = np.mean(accx)
  mn_y = np.mean(accy)
  mn_z = np.mean(accz)

  # The standard deviation of each channel
  std_x = np.std(accx)
  std_y = np.std(accy)
  std_z = np.std(accz)

  # Various percentile values for each channel
  p5_x = np.percentile(accx, 5)
  p5_y = np.percentile(accy, 5)
  p5_z = np.percentile(accz, 5)
  p10_x = np.percentile(accx, 10)
  p10_y = np.percentile(accy, 10)
  p10_z = np.percentile(accz, 10)
  p25_x = np.percentile(accx, 25)
  p25_y = np.percentile(accy, 25)
  p25_z = np.percentile(accz, 25)
  p50_x = np.percentile(accx, 50)
  p50_y = np.percentile(accy, 50)
  p50_z = np.percentile(accz, 50)
  p90_x = np.percentile(accx, 90)
  p90_y = np.percentile(accy, 90)
  p90_z = np.percentile(accz, 90)

  # The pearson correlation of all pairs of channels
  corr_xy = sp.stats.pearsonr(accx, accy)[0]
  corr_xz = sp.stats.pearsonr(accx, accz)[0]
  corr_yz = sp.stats.pearsonr(accy, accz)[0]

  # The total energy of each channel
  energy_x = np.sum(np.square(accx - np.mean(accx)))
  energy_y = np.sum(np.square(accy - np.mean(accy)))
  energy_z = np.sum(np.square(accz - np.mean(accz)))

  # Take an FFT of the signal. If the signal is too short, 0-pad it so we have 
  # at least 2046 points in the FFT.
  fft_len = max(len(accx), 2046)

  # Create an array of frequency bins
  fft_freqs = np.fft.rfftfreq(fft_len, 1 / fs)

  # Helper function to select frequency bins between <low> and <high>
  freqs_bw = lambda low, high: (fft_freqs >= low) & (fft_freqs <= high)

  # Compute the accelerometer magnitude
  accm = np.sqrt(np.sum(np.square(np.vstack((accx, accy, accz))), axis=0))

  # Take an FFT of the centered signal
  fft_x = np.fft.rfft(accx, fft_len)
  fft_y = np.fft.rfft(accy, fft_len)
  fft_z = np.fft.rfft(accz, fft_len)
  fft_m = np.fft.rfft(accm, fft_len)

  # Compute the energy spectrum
  spec_energy_x = np.square(np.abs(fft_x))
  spec_energy_y = np.square(np.abs(fft_y))
  spec_energy_z = np.square(np.abs(fft_z))
  spec_energy_m = np.square(np.abs(fft_m))

  # The frequency with the most power between 0.25 and 12 Hz
  dom_x = fft_freqs[np.argmax(fft_x[freqs_bw(0.25, 12)])]
  dom_y = fft_freqs[np.argmax(fft_y[freqs_bw(0.25, 12)])]
  dom_z = fft_freqs[np.argmax(fft_z[freqs_bw(0.25, 12)])]
  dom_m = fft_freqs[np.argmax(fft_m[freqs_bw(0.25, 12)])]

  # The fraction of energy in various frequency bins for each channel
  energy_01_x = (np.sum(spec_energy_x[freqs_bw(0, 1)]) 
                 / np.sum(spec_energy_x))
  energy_01_y = (np.sum(spec_energy_x[freqs_bw(0, 1)]) 
                 / np.sum(spec_energy_y))
  energy_01_z = (np.sum(spec_energy_x[freqs_bw(0, 1)]) 
                 / np.sum(spec_energy_z))
  energy_01_m = (np.sum(spec_energy_x[freqs_bw(0, 1)]) 
                 / np.sum(spec_energy_m))
  energy_12_x = (np.sum(spec_energy_x[freqs_bw(1, 2)]) 
                 / np.sum(spec_energy_x))
  energy_12_y = (np.sum(spec_energy_x[freqs_bw(1, 2)]) 
                 / np.sum(spec_energy_y))
  energy_12_z = (np.sum(spec_energy_x[freqs_bw(1, 2)]) 
                 / np.sum(spec_energy_z))
  energy_12_m = (np.sum(spec_energy_x[freqs_bw(1, 2)]) 
                 / np.sum(spec_energy_m))
  energy_23_x = (np.sum(spec_energy_x[freqs_bw(2, 3)]) 
                 / np.sum(spec_energy_x))
  energy_23_y = (np.sum(spec_energy_x[freqs_bw(2, 3)]) 
                 / np.sum(spec_energy_y))
  energy_23_z = (np.sum(spec_energy_x[freqs_bw(2, 3)]) 
                 / np.sum(spec_energy_z))
  energy_23_m = (np.sum(spec_energy_x[freqs_bw(2, 3)]) 
                 / np.sum(spec_energy_m))
  energy_34_x = (np.sum(spec_energy_x[freqs_bw(3, 4)]) 
                 / np.sum(spec_energy_x))
  energy_34_y = (np.sum(spec_energy_x[freqs_bw(3, 4)]) 
                 / np.sum(spec_energy_y))
  energy_34_z = (np.sum(spec_energy_x[freqs_bw(3, 4)]) 
                 / np.sum(spec_energy_z))
  energy_34_m = (np.sum(spec_energy_x[freqs_bw(3, 4)]) 
                 / np.sum(spec_energy_m))
  energy_45_x = (np.sum(spec_energy_x[freqs_bw(4, 5)]) 
                 / np.sum(spec_energy_x))
  energy_45_y = (np.sum(spec_energy_x[freqs_bw(4, 5)]) 
                 / np.sum(spec_energy_y))
  energy_45_z = (np.sum(spec_energy_x[freqs_bw(4, 5)]) 
                 / np.sum(spec_energy_z))
  energy_45_m = (np.sum(spec_energy_x[freqs_bw(4, 5)]) 
                 / np.sum(spec_energy_m))
  energy_56_x = (np.sum(spec_energy_x[freqs_bw(5, 6)]) 
                 / np.sum(spec_energy_x))
  energy_56_y = (np.sum(spec_energy_x[freqs_bw(5, 6)]) 
                 / np.sum(spec_energy_y))
  energy_56_z = (np.sum(spec_energy_x[freqs_bw(5, 6)]) 
                 / np.sum(spec_energy_z))
  energy_56_m = (np.sum(spec_energy_x[freqs_bw(5, 6)]) 
                 / np.sum(spec_energy_m))
  

  return (mn_x,
          mn_y,
          mn_z,
          std_x,
          std_y,
          std_z,
          p5_x,
          p5_y,
          p5_z,
          p10_x,
          p10_y,
          p10_z,
          p25_x,
          p25_y,
          p25_z,
          p50_x,
          p50_y,
          p50_z,
          p90_x,
          p90_y,
          p90_z,
          corr_xy,
          corr_xz,
          corr_yz,
          energy_x,
          energy_y,
          energy_z,
          dom_x,
          dom_y,
          dom_z,
          dom_m,
          energy_01_x,
          energy_12_x,
          energy_23_x,
          energy_34_x,
          energy_45_x,
          energy_56_x,
          energy_01_y,
          energy_12_y,
          energy_23_y,
          energy_34_y,
          energy_45_y,
          energy_56_y,
          energy_01_z,
          energy_12_z,
          energy_23_z,
          energy_34_z,
          energy_45_z,
          energy_56_z,
          energy_01_m,
          energy_12_m,
          energy_23_m,
          energy_34_m,
          energy_45_m,
          energy_56_m,
          )

def FeatureNames():
  """Returns the names of all the features."""
  return ('mn_x',
          'mn_y',
          'mn_z',
          'std_x',
          'std_y',
          'std_z',
          'p5_x',
          'p5_y',
          'p5_z',
          'p10_x',
          'p10_y',
          'p10_z',
          'p25_x',
          'p25_y',
          'p25_z',
          'p50_x',
          'p50_y',
          'p50_z',
          'p90_x',
          'p90_y',
          'p90_z',
          'corr_xy',
          'corr_xz',
          'corr_yz',
          'energy_x',
          'energy_y',
          'energy_z',
          'dom_x',
          'dom_y',
          'dom_z',
          'dom_m',
          'energy_01_x',
          'energy_12_x',
          'energy_23_x',
          'energy_34_x',
          'energy_45_x',
          'energy_56_x',
          'energy_01_y',
          'energy_12_y',
          'energy_23_y',
          'energy_34_y',
          'energy_45_y',
          'energy_56_y',
          'energy_01_z',
          'energy_12_z',
          'energy_23_z',
          'energy_34_z',
          'energy_45_z',
          'energy_56_z',
          'energy_01_m',
          'energy_12_m',
          'energy_23_m',
          'energy_34_m',
          'energy_45_m',
          'energy_56_m',
          )


def PlotConfusionMatrix(cm, classes,
                       normalize=False,
                       title=None,
                       cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig = plt.figure(figsize=(8, 5))
    im = plt.imshow(cm, interpolation='nearest', cmap=cmap)
    ax = fig.get_axes()[0]
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           xlim=(-.5, len(classes) - .5),
           ylim=(len(classes) - .5, -.5),
           title=title,
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
    return ax


def GenerateFeatures(data, fs, window_length_s, window_shift_s):
  """Extract features from the dataset.

  Generate features by sliding a window across each dataset and computing
  the features on each window.

  Args:
    data: (list) As returned by LoadWristPPGDataset()
    fs: (number) The sampling rate of the data
    window_length_s: (number) The length of the window in seconds
    window_shift_s: (number) The amount to shift the window by

  Returns:
    A 3-tuple:
    labels: (np.array) Class labels
    subjects: (np.array) The subject id that the datapoint came from
    features: (np.array) 2D Array, n_samples X n_features. The feature matrix.
  """
  window_length = window_length_s * fs
  window_shift = window_shift_s * fs
  labels, subjects, features = [], [], []
  for subject, activity, df in data:
    for i in range(0, len(df) - window_length, window_shift):
      window = df[i: i + window_length]
      accx = window.accx.values
      accy = window.accy.values
      accz = window.accz.values
      features.append(Featurize(accx, accy, accz, fs=fs))
      labels.append(activity)
      subjects.append(subject)
  labels = np.array(labels)
  subjects = np.array(subjects)
  features = np.array(features)
  return labels, subjects, features


def LOSOCVPerformance(features, labels, subjects, clf):
  """Return the confusion matrix for the classifier.

  Using leave-one-subject-out cross validation.

  Args:
    features: (np.array) 2D Array, n_samples X n_features. The feature matrix.
    labels: (np.array) Class labels
    subjects: (np.array) The subject id that the datapoint came from
    clf: (sklearn classifier) The model to evaluate.

  Returns:
    A 3x3 confusion matrix.
  """
  class_names = ['bike', 'walk', 'run']
  logo = LeaveOneGroupOut()
  cm = np.zeros((3, 3), dtype='int')
  for train_ind, test_ind in logo.split(features, labels, subjects):
    X_train, y_train = features[train_ind], labels[train_ind]
    X_test, y_test = features[test_ind], labels[test_ind]
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    c = confusion_matrix(y_test, y_pred, labels=class_names)
    cm += c
  return cm

