import sys
from moviepy.editor import VideoFileClip
import numpy as np
import scipy.misc
import tensorflow as tf
from helper import annotate_image

def convert_video(pipeline, path_in, path_out):
    clip_in = VideoFileClip(path_in)
    clip_out = clip_in.fl_image(pipeline)
    clip_out.write_videofile(path_out, audio=False)

def main(sess_path, path_in, path_out):
    aspect_ratio = np.array([9, 16])
    min_pow2 = 32 # output height and width each needs to be divisible by this
    out_resolution = aspect_ratio * min_pow2

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(f"{sess_path}.meta")
        saver.restore(sess, sess_path)

        logits, keep_prob, image_input = (
            tf.get_collection(n)[0] for n in [
                'logits', 'keep_prob', 'image_input'
            ])
        print(logits, keep_prob, image_input)

        def pipeline(img_in):
            img_in = scipy.misc.imresize(img_in, out_resolution)
            return annotate_image(sess, logits, keep_prob, image_input, img_in)

        convert_video(pipeline, path_in, path_out)

if __name__ == '__main__':
    argv = sys.argv[1:]
    if len(argv) != 3:
        print('arguments are [sess_path, path_in, path_out]')
        exit(1)
    main(*argv)
