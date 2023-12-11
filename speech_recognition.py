# %% [markdown]
# # Speech Recognition
#
# ## Requirements

# %%
#!pip install -q pandas numpy matplotlib scipy scikit-learn librosa

# %%
# Imports
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa
import python_speech_features as speech
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture

# %% [markdown]
# ## Data processing
# 
# Download and extract samples from [**here**](https://cloud.minesparis.psl.eu/index.php/s/UwiQksQwLbDWkMO). A folder named `dc-corpus` should be created with 2 subfolders. If it's not the case, don't forget to adjust the path to the extracted folder (first line of the cell below).
# 
# The following code loads 10 audio files for each speaker, cuts silence at the beginning and end of the file (power<30 dB). You can also add real noise to the current track (street, neighborhood, crowd, etc) by uncommenting the mixing line (`y = 0.8 * y ...`). Finally, compute the features of the input signal and create the dataset.

# %%
DATA_DIR = Path('dc-corpus')
AUDIO_DIR = DATA_DIR / 'audio'
NOISE_DIR = DATA_DIR / 'noise'

# For reproducibility 
seed = 42
rng = np.random.default_rng(seed)

sounds = sorted(AUDIO_DIR.glob('**/*.mp3'))
noises = rng.choice(sorted(NOISE_DIR.glob('**/*.mp3')), len(sounds))

data = []
for mp3file, noisefile in zip(sounds, noises):
    # Load file
    speaker = mp3file.relative_to(AUDIO_DIR).parent
    audio, rate = librosa.load(mp3file)
    duration = len(audio) / rate

    # Preprocessing
    y, (start, end) = librosa.effects.trim(audio, top_db=30)

    # Add noise to test robustness
    n, _ = librosa.load(noisefile, duration=(end-start)/rate, offset=start/rate, sr=rate)
    # y = 0.8 * y + 0.2 * n  # mix background noise

    # Postprocessing
    y = librosa.effects.preemphasis(y)

    # Compute audio features (MFCC + Delta)
    mfcc = speech.mfcc(y, rate, winlen=0.025, winstep=0.01, numcep=20, nfft=2048, winfunc=np.hamming)
    mfcc = scale(mfcc)
    delta = speech.delta(mfcc, 2)
    delta2 = speech.delta(delta, 2)
    features = np.hstack((mfcc, delta, delta2))

    # Collect data
    data.append({'speaker': speaker, 'path': mp3file, 'duration': duration,
                 'rate': rate, 'audio': audio, 'features': features})

# Build dataset
df = pd.DataFrame(data)

# %% [markdown]
# # Modelisation
# 
# Now, split the dataset in two parts for training and testing. Aggregate all sentences from a speaker and compute a Gaussian mixture model (like a K-Mean but probabilistic) on train data.

# %%
# Split dataset
X_train, X_test = train_test_split(df, test_size=0.5, stratify=df['speaker'], random_state=seed)

def process_gaussian_mixture(features):
    features = np.vstack(features)
    gmm = GaussianMixture(n_components=16, covariance_type='diag',
                          max_iter=200, n_init=3, random_state=seed)
    return gmm.fit(features)

# Build Gaussian mixture models
models = {speaker: process_gaussian_mixture(features)
            for speaker, features in X_train.groupby('speaker')['features']}

# %% [markdown]
# ## Evaluation

# %%
# Test model
results = []
for row in X_test.itertuples():
    y_prob = [gmm.score(row.features) for speaker, gmm in models.items()]
    results.append(y_prob)

out = pd.DataFrame(results, index=X_test['speaker'], columns=models.keys())
out.insert(0, 'Speaker', out.idxmax(axis=1))
print(f'Log Likelihood Average: {out.Speaker.eq(out.index).mean():.3f}')
display(pd.crosstab(out.index, out.Speaker, rownames=['Real'], colnames=['Pred'])
          .reindex(columns=models.keys(), fill_value=0))

# %% [markdown]
# ## References
# 
# - [Speaker Identification Using GMM with MFCC](https://www.researchgate.net/profile/Tahira-Mahboob/publication/274963749_Speaker_Identification_Using_GMM_with_MFCC/links/5641907f08aeacfd89366844/Speaker-Identification-Using-GMM-with-MFCC.pdf)
# - [Automatic Speaker Recognition System Based on Gaussian
# Mixture Models, Cepstral Analysis, \[...\]](https://mdpi-res.com/d_attachment/sensors/sensors-22-09370/article_deploy/sensors-22-09370.pdf)
# - [MFCC AND DTW BASED SPEECH RECOGNITION](https://www.irjet.net/archives/V5/i2/IRJET-V5I2408.pdf)
# - [Speaker Verification System Using MFCC and DWT](https://www.iosrjournals.org/iosr-jece/papers/wicomet-volume1/E.pdf)
# - [Python Speech Features](https://github.com/jameslyons/python_speech_features)
