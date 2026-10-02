# Licensed under the Apache License, Version 2.0.
# See LICENSE for the full license text.

"""Dual-channel Mel+CQT conditioning, saved as [2, 80, frames]."""

import librosa
import numpy as np
import torch
import torchaudio as T
import torchaudio.transforms as TT

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
  mel_args = {
      'sample_rate': sr,
      'win_length': params.hop_samples * 4,
      'hop_length': params.hop_samples,
      'n_fft': params.n_fft,
      'f_min': 20.0,
      'f_max': sr / 2.0,
      'n_mels': params.n_mels,
      'power': 1.0,
      'normalized': True,
      'pad_mode': 'reflect' if audio.numel() > params.n_fft // 2 else 'constant',
  }
  mel_spec_transform = TT.MelSpectrogram(**mel_args)

  with torch.no_grad():
    mel_spectrogram = mel_spec_transform(audio)
    mel_spectrogram = 20 * torch.log10(torch.clamp(mel_spectrogram, min=1e-5)) - 20
    mel_spectrogram = torch.clamp((mel_spectrogram + 100) / 100, 0.0, 1.0)
    # The paper does not specify f_min; use librosa's C1 default explicitly.
    cqt = librosa.cqt(
        audio.cpu().numpy(), sr=sr, hop_length=params.hop_samples,
        fmin=librosa.note_to_hz('C1'), n_bins=params.n_mels,
        bins_per_octave=12, pad_mode='constant')
    cqt_spectrogram = torch.from_numpy(np.abs(cqt).astype(np.float32))
    cqt_spectrogram = 20 * torch.log10(torch.clamp(cqt_spectrogram, min=1e-5)) - 20
    cqt_spectrogram = torch.clamp((cqt_spectrogram + 100) / 100, 0.0, 1.0)

    if mel_spectrogram.shape != cqt_spectrogram.shape:
      raise ValueError(f'Mel/CQT frame mismatch for {filename}: '
          f'{mel_spectrogram.shape} vs {cqt_spectrogram.shape}.')
    stack_spectrogram = torch.stack([mel_spectrogram, cqt_spectrogram], dim=0)
    np.save(f'{filename}.spec.npy', stack_spectrogram.cpu().numpy())


def main(args):
  filenames = glob(f'{args.dir}/**/*.wav', recursive=True)
  with ProcessPoolExecutor() as executor:
    list(tqdm(executor.map(transform, filenames), desc='Preprocessing', total=len(filenames)))


if __name__ == '__main__':
  parser = ArgumentParser(description='prepares a dataset to train DiffWave')
  parser.add_argument('dir',
      help='directory containing .wav files for training')
  main(parser.parse_args())
