# nsynth notification sound generation

scripts to generate phone message notification sounds nsynth by magenta.

## setup

1. to create the envorinment: `conda env create -f environment.yml`
2. activate the enviroment: `conda activate nsynth`

# sound generation

i used a pretrained wavenet model by magenta (can be downloaded here: http://download.magenta.tensorflow.org/models/nsynth/wavenet-ckpt.tar). then i recorded different notification sounds from my phone, tweaked the volume so they are similarly loud and cut them to the same length. i picked six snappy short ones in the end. with these six as an input i can generate as many different abstractions with `generate_notifications.py` as i like. they all sound kind of similar but all of them are a little different
