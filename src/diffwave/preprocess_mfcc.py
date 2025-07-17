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
    mfcc_transform = TT.MFCC(
    sample_rate=sr,
    n_mfcc=84,              # Number of MFCC coefficients
    melkwargs={
        'n_fft': 1024,
        'hop_length': 256,
        'n_mels': 128,
        'f_min': 20.0,
        'f_max': sr / 2.0,
        'win_length': 256 * 4,
        'power': 2.0,
        'normalized': True,
    }
    )
    with torch.no_grad():
        mfcc = mfcc_transform(audio)
        mfcc = 20 * torch.log10(torch.clamp(mfcc, min=1e-5)) - 20
        mfcc = torch.clamp((mfcc + 125) / 125, 0.0, 1.0)
        np.save(f'{filename}.spec.npy', mfcc.cpu().numpy())


def main(args):
  filenames = glob(f'{args.dir}/**/*.wav', recursive=True)
  with ProcessPoolExecutor() as executor:
    list(tqdm(executor.map(transform, filenames), desc='Preprocessing', total=len(filenames)))


if __name__ == '__main__':
  parser = ArgumentParser(description='prepares a dataset to train DiffWave')
  parser.add_argument('dir',
      help='directory containing .wav files for training')
  main(parser.parse_args())
