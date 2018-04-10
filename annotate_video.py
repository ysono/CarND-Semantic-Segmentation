import sys
from moviepy.editor import VideoFileClip
import scipy.misc
import tensorflow as tf
from helper import annotate_image

def convert_video(pipeline, path_in, path_out):
    clip_in = VideoFileClip(path_in)
    clip_out = clip_in.fl_image(pipeline)
    clip_out.write_videofile(path_out, audio=False)

def main(sess_path, path_in, path_out):
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(f"{sess_path}.meta")
        saver.restore(sess, sess_path)

        logits, keep_prob, image_input = (
            tf.get_collection(n)[0] for n in [
                'logits', 'keep_prob', 'image_input'
            ])
        print(logits, keep_prob, image_input)

        image_shape = (int(720*2/5), int(1280*2/5))
        def pipeline(img_in):
            img_in = scipy.misc.imresize(img_in, image_shape)
            return annotate_image(sess, logits, keep_prob, image_input, img_in)

        convert_video(pipeline, path_in, path_out)

if __name__ == '__main__':
    argv = sys.argv[1:]
    if len(argv) != 3:
        print('arguments are [sess_path, path_in, path_out]')
        exit(1)
    main(*argv)
