from typing import List
import os
import glob
import random
import click
import numpy as np
from magenta.models.nsynth import utils
from magenta.models.nsynth.wavenet import fastgen
from skimage.transform import resize
from pydub import AudioSegment


# timestretching helper function
def timestretch(encodings, factor):
    min_encoding, max_encoding = encodings.min(), encodings.max()
    encodings_norm = (encodings - min_encoding) / (max_encoding - min_encoding)
    timestretches = []
    for encoding_i in encodings_norm:
        stretched = resize(
            encoding_i,
            (int(encoding_i.shape[0] * factor), encoding_i.shape[1]),
            mode='reflect')
        stretched = (stretched * (max_encoding - min_encoding)) + min_encoding
        timestretches.append(stretched)
    return np.array(timestretches)


# audio speed change function
def speed_change(sound, speed=1.0):
    # Manually override the frame_rate. This tells the computer how many
    # samples to play per second
    sound_with_altered_frame_rate = sound._spawn(
        sound.raw_data,
        overrides={"frame_rate": int(sound.frame_rate * speed)})

    # convert the sound with altered frame rate to a standard frame rate
    # so that regular playback programs will work right. They often only
    # know how to play audio at standard frame rate (like 44.1k)
    return sound_with_altered_frame_rate.set_frame_rate(sound.frame_rate)


def parse_deviation(s: str) -> List[float]:
    return [float(sub.strip()) for sub in s.split(',')]


# yapf: disable
@click.command()
@click.option('--sample_rate', type=int, default=16000, help='sample rate of the input audio files')
@click.option('--length', type=float, default=5, help='length of the input audio files')
@click.option('--batch_size', type=int, default=10, help='how many fused sounds to encode in one batch')
@click.option('--executions', type=int, default=10, help='how many batches to process')
@click.option('--in_dir', type=click.Path(exists=True, file_okay=False), help='directory where the input files are at')
@click.option('--out_dir', type=click.Path(exists=False, file_okay=False), help='where to put the output files')
@click.option('--deviation', type=parse_deviation, default=[0.9, 1.1], required=True, help='random deviation to vary the fused sounds')
# yapf: enable
def generate(sample_rate: int, length: float, batch_size: int, executions: int,
             in_dir: str, out_dir: str, deviation: List[float]) -> None:
    sample_length = int(sample_rate * length)

    # load audio files into numpy arrays
    audios = []
    for path in glob.glob(os.path.join(in_dir, '*.wav')):

        audio = utils.load_audio(path,
                                 sample_length=sample_length,
                                 sr=sample_rate)
        print(
            f'loaded {path}. samples: {audio.shape[0]}. expected samples: {sample_length}'
        )
        audios.append(audio)

    # encode
    encodings = [
        fastgen.encode(
            audio, os.path.join('models', 'wavenet-ckpt', 'model.ckpt-200000'),
            sample_length) for audio in audios
    ]

    for execution_count, _ in enumerate(range(executions)):
        # fuse encodings randomly
        fused_encodings_list = []
        for _ in range(batch_size):
            # shuffle the list in the beginning
            random.shuffle(encodings)

            # function for getting a factor for a random proportion
            # e.g. if there are 4 encodings, each would have a proportional factor of 1/4=0,25
            # this fuction randomizes this factor with a minimum and maximum devation
            def random_proportion(count: int, deviation_min: float,
                                  deviation_max: float) -> float:
                return 1 / count * random.uniform(deviation_min, deviation_max)

            # total proportion
            remaining = 1

            fused_encoding = np.full_like(encodings[0], 0)
            for index, encoding in enumerate(encodings):
                proportion = random_proportion(
                    len(encodings), deviation[0],
                    deviation[1])  # calculate the proportion

                if proportion > remaining:
                    proportion = remaining
                    remaining = 0
                elif index == len(encodings) - 1:
                    proportion = remaining

                fused_encoding += encoding * proportion  # add the proportion to total average

                remaining -= proportion  # subtract proportion from total proportion

            fused_encodings_list.append(fused_encoding)

        # concat all fused encodings in numpy array for parallel processing
        fused_encodings = np.concatenate(fused_encodings_list, axis=0)

        fused_encodings = timestretch(fused_encodings, 0.8)

        save_paths = [
            os.path.join(out_dir,
                         f'fused_{execution_count * batch_size + i}.wav')
            for i in range(fused_encodings.shape[0])
        ]

        # generate batchwise
        fastgen.synthesize(fused_encodings,
                           save_paths=save_paths,
                           checkpoint_path=os.path.join(
                               'models', 'wavenet-ckpt', 'model.ckpt-200000'),
                           samples_per_save=sample_length)

        # reopen the wav files, stretch them & save them as mp3s
        for path in save_paths:
            speed_change(AudioSegment.from_wav(path),
                         0.8).export(path.replace('wav', 'mp3'),
                                     format='mp3',
                                     parameters=['-q:a', '9'])


if __name__ == '__main__':
    generate()