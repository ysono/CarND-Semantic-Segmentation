import sys
import tensorflow as tf
from helper import save_inference_samples

def run(sess_path):
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(f"{sess_path}.meta")
        saver.restore(sess, sess_path)

        logits, keep_prob, image_input = (
            tf.get_collection(n)[0] for n in [
                'logits', 'keep_prob', 'image_input'
            ])
        print(logits, keep_prob, image_input)
    
        save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, image_input)

if __name__ == '__main__':
    argv = sys.argv[1:]
    if len(argv) != 1:
        print('arguments are [sess_path]')
        exit(1)
    run(argv[0])
