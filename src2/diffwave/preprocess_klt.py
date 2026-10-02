# Licensed under the Apache License, Version 2.0.
# See LICENSE for the full license text.

"""Two KLT component maps, saved as [2, 80, frames].

Section IV-F specifies an 80-dimensional STFT basis and two eigenvectors,
but not the frequency reduction, covariance scope, or channel construction.
Here, STFT magnitudes are averaged into 80 linear frequency bands, a centered
covariance is fitted per recording, and each retained projection is mapped
back to its rank-one contribution in the 80-band spectral space. These are
implementation choices, not a claim of exact experimental reproduction.
The mean is not added back to the two component maps. Signed compression
preserves negative KLT values (zero maps to 0.5).
"""

import numpy as np
import torch
import torchaudio as T
import torchaudio.transforms as TT

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from glob import glob
from tqdm import tqdm

from diffwave.params import params


def klt_components(spectrogram):
  # Average linear STFT bins to the fixed 80-sample basis used in the paper.
  bands = np.stack([
      band.mean(axis=0)
      for band in np.array_split(spectrogram, params.n_mels, axis=0)
  ]).astype(np.float64)
  centered = bands - bands.mean(axis=1, keepdims=True)
  covariance = centered @ centered.T / max(centered.shape[1] - 1, 1)
  eigenvalues, eigenvectors = np.linalg.eigh(covariance)
  basis = eigenvectors[:, np.argsort(eigenvalues)[::-1][:2]]
  # Fix the arbitrary sign of eigenvectors for repeatable projection scores.
  for index in range(2):
    pivot = np.argmax(np.abs(basis[:, index]))
    if basis[pivot, index] < 0:
      basis[:, index] *= -1
  scores = basis.T @ centered
  components = np.einsum('fk,kt->kft', basis, scores)
  # Unlike a magnitude spectrogram, KLT component maps can be negative.
  components = np.sign(components) * np.log1p(np.abs(components))
  scale = np.max(np.abs(components), axis=(1, 2), keepdims=True)
  components = 0.5 + 0.5 * components / np.maximum(scale, 1e-8)
  return np.clip(components, 0.0, 1.0).astype(np.float32)


def transform(filename):
  audio, sr = T.load(filename)
  audio = torch.clamp(audio[0], -1.0, 1.0)

  if params.sample_rate != sr:
    raise ValueError(f'Invalid sample rate {sr}.')

  spec_args = {
      'win_length': params.hop_samples * 4,
      'hop_length': params.hop_samples,
      'n_fft': params.n_fft,
      'power': 1.0,
      'normalized': True,
      'pad_mode': 'reflect' if audio.numel() > params.n_fft // 2 else 'constant',
  }
  spec_transform = TT.Spectrogram(**spec_args)

  with torch.no_grad():
    spectrogram = spec_transform(audio).cpu().numpy()
    components = klt_components(spectrogram)
    np.save(f'{filename}.spec.npy', components)


def main(args):
  filenames = glob(f'{args.dir}/**/*.wav', recursive=True)
  with ProcessPoolExecutor() as executor:
    list(tqdm(executor.map(transform, filenames), desc='Preprocessing', total=len(filenames)))


if __name__ == '__main__':
  parser = ArgumentParser(description='prepares a dataset to train DiffWave')
  parser.add_argument('dir',
      help='directory containing .wav files for training')
  main(parser.parse_args())
