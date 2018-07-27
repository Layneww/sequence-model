# StreetView Tensorflow Recurrent End-to-End Transcription (STREET) Model.

Refer to 
A TensorFlow implementation of the STREET model described in the paper:

"End-to-End Interpretation of the French Street Name Signs Dataset"

Raymond Smith, Chunhui Gu, Dar-Shyang Lee, Huiyi Hu, Ranjith
Unnikrishnan, Julian Ibarz, Sacha Arnoud, Sophia Lin.

*International Workshop on Robust Reading, Amsterdam, 9 October 2016.*

Available at: http://link.springer.com/chapter/10.1007%2F978-3-319-46604-0_30

## Contents
* [Introduction](#introduction)
* [Installing and setting up the STREET model](#installing-and-setting-up-the-street-model)
* [Confidence Tests](#confidence-tests)
* [Training a model](#training-a-model)
* [The Variable Graph Specification Language](#the-variable-graph-specification-language)

## Introduction

The model trains both Thai and English in one charset.


## Installing and setting up the STREET model
[Install Tensorflow](https://www.tensorflow.org/install/)

Install numpy:

```
sudo pip install numpy
```

Build the LSTM op:
```
cd cc
TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
g++ -std=c++11 -shared rnn_ops.cc -o rnn_ops.so -fPIC -I $TF_INC -O3 -mavx -D_GLIBCXX_USE_CXX11_ABI=0 -L $TF_LIB -ltensorflow_framework -O2
```
If appearing ```fatal error: nsync_cv.h: No such a file or directory```, add ```-I$TF_INC/external/nsync/public``` to the g++ command.
Refering to [TensorFlow Instruction](https://www.tensorflow.org/extend/adding_an_op#build_the_op_library) for more details.

Build tesseract:
Refer to https://github.com/Layneww/Tesseract-Notes/blob/master/setupTesseract.md, Build Tesseract from source.

Run the unittests:

```
cd ../python
python decoder_test.py
python errorcounter_test.py
python shapes_test.py
python vgslspecs_test.py
python vgsl_model_test.py
```

## Training a full model
```
cd python
train_dir=../model
# rm -rf $train_dir
CUDA_VISIBLE_DEVICES="0" python3 vgsl_train.py -model_str='1,60,0,1[Ct5,5,16 Mp3,3 Lfys64 Lfx128 Lrx128 Lfx256]O1c225' --train_data=../dataset_ctc/tha+eng/tfrecords/train/* --train_dir=$train_dir --max_steps=1000000 &
CUDA_VISIBLE_DEVICES="" python3 vgsl_eval.py --model_str='1,60,0,1[Ct5,5,16 Mp3,3 Lfys64 Lfx128 Lrx128 Lfx256]O1c225' --num_steps=1000 --eval_data=../dataset_ctc/tha+eng/tfrecords/eval/*  --decoder=../dataset_ctc/tha+eng/charset_size=225.txt --eval_interval_secs=300 --train_dir=$train_dir --eval_dir=$train_dir/eval &
tensorboard --logdir=$train_dir
```

## The Variable Graph Specification Language

The STREET model makes use of a graph specification language (VGSL) that
enables rapid experimentation with different model architectures. The language
defines a Tensor Flow graph that can be used to process images of variable sizes
to output a 1-dimensional sequence, like a transcription/OCR problem, or a
0-dimensional label, as for image identification problems. For more information
see [vgslspecs](g3doc/vgslspecs.md)
