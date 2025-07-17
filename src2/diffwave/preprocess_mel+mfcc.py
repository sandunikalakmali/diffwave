# Copyright 2020 LMNT, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

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

    #   if params.sample_rate != sr:
    #     raise ValueError(f'Invalid sample rate {sr}.')
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
    }
    mel_spec_transform = TT.MelSpectrogram(**mel_args)

    mfcc_transform = TT.MFCC(
    sample_rate=sr,
    n_mfcc=params.n_mels,              # Number of MFCC coefficients
    melkwargs={
        'n_fft': params.n_fft,
        'hop_length': params.hop_samples,
        'n_mels': 128,
        'f_min': 20.0,
        'f_max': sr / 2.0,
        'win_length': params.hop_samples * 4,
        'power': 2.0,
        'normalized': True,
    }
    )

    with torch.no_grad():
        mel_spectrogram = mel_spec_transform(audio)
        mel_spectrogram = 20 * torch.log10(torch.clamp(mel_spectrogram, min=1e-5)) - 20
        mel_spectrogram = torch.clamp((mel_spectrogram + 100) / 100, 0.0, 1.0)

        mfcc = mfcc_transform(audio)
        mfcc = 20 * torch.log10(torch.clamp(mfcc, min=1e-5)) - 20
        mfcc = torch.clamp((mfcc + 125) / 125, 0.0, 1.0)

        stack_spectrogram = torch.stack([mel_spectrogram, mfcc], dim=0)
        np.save(f'{filename}.spec.npy', stack_spectrogram.numpy())


def main(args):
  filenames = glob(f'{args.dir}/**/*.wav', recursive=True)
  with ProcessPoolExecutor() as executor:
    list(tqdm(executor.map(transform, filenames), desc='Preprocessing', total=len(filenames)))


if __name__ == '__main__':
  parser = ArgumentParser(description='prepares a dataset to train DiffWave')
  parser.add_argument('dir',
      help='directory containing .wav files for training')
  main(parser.parse_args())
