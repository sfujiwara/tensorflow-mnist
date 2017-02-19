# -*- coding: utf-8 -*-

import argparse
import json
import os

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import modelsr
import modelcnn

tf.logging.set_verbosity(tf.logging.DEBUG)

parser = argparse.ArgumentParser()
parser.add_argument("--output_path", type=str)
parser.add_argument("--algorithm", type=str, default="cnn")
args, unknown_args = parser.parse_known_args()
tf.logging.info("known args: {}".format(args))

# Get environment variable for Cloud ML
tf_conf = json.loads(os.environ.get("TF_CONFIG", "{}"))
# For local
if not tf_conf:
    tf_conf = {
      "cluster": {"master": ["localhost:2222"]},
      "task": {"index": 0, "type": "master"}
    }
tf.logging.debug("TF_CONF: {}".format(json.dumps(tf_conf)))

# Cluster setting for cloud
cluster = tf_conf.get("cluster", None)


def main(_):
    # Select model (Softmax Regression or CNN)
    if args.algorithm == "softmax_regression":
        tf.logging.info("algorithm: softmax regression")
        model = modelsr.MnistSr()
    else:
        tf.logging.info("algorithm: convolutional neural network")
        model = modelcnn.MnistCnn()

    cluster_spec = tf.train.ClusterSpec(cluster=cluster)
    server = tf.train.Server(
        cluster,
        job_name=tf_conf["task"]["type"],
        task_index=tf_conf["task"]["index"]
    )

    # Parameter server
    if tf_conf["task"]["type"] == "ps":
        server.join()
    # Master and workers
    else:
        device_fn = tf.train.replica_device_setter(
            cluster=cluster_spec,
            worker_device="/job:{0}/task:{1}".format(tf_conf["task"]["type"], tf_conf["task"]["index"]),
        )
        # Build graph
        tf.logging.debug("/job:{0}/task:{1} build graph".format(tf_conf["task"]["type"], tf_conf["task"]["index"]))
        tf.logging.debug(tf.get_default_graph().get_operations())
        with tf.Graph().as_default() as graph:
            with tf.device(device_fn):
                global_step = tf.Variable(0, trainable=False, name="global_step")
                x_ph = tf.placeholder(tf.float32, shape=[None, 784], name="x_ph")
                y_ph = tf.placeholder(tf.float32, shape=[None, 10], name="y_ph")
                keep_prob_ph = tf.placeholder(tf.float32, name="keep_prob_ph")
                logits = model.inference(x_ph, is_training=True, keep_prob_ph=keep_prob_ph)
                loss = model.build_loss(y_ph, logits)
                tf.scalar_summary("loss", loss)
                train_op = tf.train.AdamOptimizer(1e-4).minimize(
                    loss,
                    global_step=global_step
                )
                with tf.name_scope("accuracy"):
                    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_ph, 1))
                    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
                    tf.scalar_summary("accuracy", accuracy)
                # Variable initializer
                init_op = tf.initialize_all_variables()
                # Summary operation
                summary_op = tf.merge_all_summaries()
                # Model saver
                # saver = tf.train.Saver()

        sv = tf.train.Supervisor(
            graph=graph,
            is_chief=(tf_conf["task"]["type"] == "master"),
            logdir=args.output_path,
            init_op=init_op,
            global_step=global_step,
            summary_op=None
        )

        with sv.managed_session(server.target) as sess:
            mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
            for i in range(1000):
                x_batch, y_batch = mnist.train.next_batch(50)
                fd = {x_ph: x_batch, y_ph: y_batch, keep_prob_ph: 0.5}
                iter, _ = sess.run([global_step, train_op], feed_dict=fd)
                # Write summary if server is master
                if tf_conf["task"]["type"] == "master" and i % 10 == 0:
                    fd = {x_ph: mnist.test.images, y_ph: mnist.test.labels, keep_prob_ph: 1.}
                    sv.summary_computed(sess, sess.run(summary_op, feed_dict=fd), global_step=i)
                    tf.logging.info(
                        "save scalar summary (iter: {0} type: {1} index: {2})".format(
                            i, tf_conf["task"]["type"], tf_conf["task"]["index"]
                        )
                    )
                # summary_writer.add_summary(summary_str, i)
                if iter % 100 == 0:
                    fd = {x_ph: mnist.test.images, y_ph: mnist.test.labels, keep_prob_ph: 1.}
                    acc = sess.run(accuracy, feed_dict=fd)
                    tf.logging.info("global_step: {}".format(iter))
                    tf.logging.info(
                        "- /job:{0}/task:{1} - Accuracy: {2}".format(
                            tf_conf["task"]["type"], tf_conf["task"]["index"], acc
                        )
                    )
            # Only master exports prediction graph
            if tf_conf["task"]["type"] == "master":
                with tf.Graph().as_default():
                    tf.logging.info("save model to {}".format(args.output_path))
                    x = tf.placeholder(tf.float32, shape=[None, 784], name="x_ph")
                    p = model.inference(x, is_training=False)
                    # Define key element
                    input_key = tf.placeholder(tf.int64, [None, ], name="key")
                    output_key = tf.identity(input_key)
                    # Define API inputs/outpus object
                    inputs = {"key": input_key.name, "image": x.name}
                    outputs = {"key": output_key.name, "scores": p.name}
                    tf.add_to_collection("inputs", json.dumps(inputs))
                    tf.add_to_collection("outputs", json.dumps(outputs))
                    # Save model
                    saver = tf.train.Saver()
                    saver.export_meta_graph(filename="{}/model/export.meta".format(args.output_path))
                    # tf.train.Saver().export_meta_graph(filename="{}/model/export.meta".format(args.output_path))
                    saver.save(sess, "{}/model/export".format(args.output_path), write_meta_graph=False)
            sv.stop()
            # if tf_conf["task"]["type"] == "master":
            #     sv.request_stop()
            # else:
            #     sv.stop()

if __name__ == '__main__':
    tf.app.run()
