"""Model inference separate from training."""
from tensorflow import app
from tensorflow.python.platform import flags
import os

import vgsl_model

flags.DEFINE_string('inf_dir', '/tmp/mdir/inf',
                    'Directory where to write event logs.')
flags.DEFINE_string('graph_def_file', None,
                    'Output eval graph definition file.')
flags.DEFINE_string('train_dir', '/tmp/mdir',
                    'Directory where to find training checkpoints.')
flags.DEFINE_string('model_str',
                    '1,60,0,1[Ct5,5,16 Mp3,3 Lfys64 Lfx128 Lrx128 Lfx256]O1c225',
                    'Network description.')
flags.DEFINE_string('infer_data', None, 'Inference data filepattern')
flags.DEFINE_string('decoder', None, 'Charset decoder')

FLAGS = flags.FLAGS


def main(argv):
  del argv
  num_lines = int(os.path.basename(FLAGS.infer_data).split('-', 1)[0])
  vgsl_model.Inference(FLAGS.train_dir, FLAGS.model_str,
                  FLAGS.infer_data, FLAGS.decoder, num_lines,
                  FLAGS.graph_def_file)

if __name__ == '__main__':
  app.run()

