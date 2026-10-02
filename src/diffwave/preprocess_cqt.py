# Licensed under the Apache License, Version 2.0.
# See LICENSE for the full license text.

"""Paper section IV-F: 80 CQT bins, 12 bins/octave, hop 256."""

import librosa
import numpy as np
import torch
import torchaudio as T

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from glob import glob
from tqdm import tqdm

from diffwave.params import params


def transform(filename):
  audio, sr = T.load(filename)
  audio = torch.clamp(audio[0], -1.0, 1.0)

  if params.sample_rate != sr:
    raise ValueError(f'Invalid sample rate {sr}.')

  with torch.no_grad():
    # The paper does not specify f_min; use librosa's C1 default explicitly.
    cqt = librosa.cqt(
        audio.cpu().numpy(), sr=sr, hop_length=params.hop_samples,
        fmin=librosa.note_to_hz('C1'), n_bins=params.n_mels,
        bins_per_octave=12, pad_mode='constant')
    cqt_spectrogram = torch.from_numpy(np.abs(cqt).astype(np.float32))
    cqt_spectrogram = 20 * torch.log10(torch.clamp(cqt_spectrogram, min=1e-5)) - 20
    cqt_spectrogram = torch.clamp((cqt_spectrogram + 100) / 100, 0.0, 1.0)
    np.save(f'{filename}.spec.npy', cqt_spectrogram.cpu().numpy())


def main(args):
  filenames = glob(f'{args.dir}/**/*.wav', recursive=True)
  with ProcessPoolExecutor() as executor:
    list(tqdm(executor.map(transform, filenames), desc='Preprocessing', total=len(filenames)))


if __name__ == '__main__':
  parser = ArgumentParser(description='prepares a dataset to train DiffWave')
  parser.add_argument('dir',
      help='directory containing .wav files for training')
  main(parser.parse_args())

