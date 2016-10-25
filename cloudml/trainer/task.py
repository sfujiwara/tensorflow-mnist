# -*- coding: utf-8 -*-

import argparse
import json
import logging
import os
import yaml

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

logging.basicConfig(level=logging.DEBUG)

parser = argparse.ArgumentParser()
parser.add_argument("--output_path", type=str)
args, unknown_args = parser.parse_known_args()
logging.info("known args: {}".format(args))

# Get environment variable for Cloud ML
tf_conf = json.loads(os.environ.get('TF_CONFIG', '{}'))
# Local
if not tf_conf:
    with open("local.yaml") as f:
        tf_conf = yaml.load(f)
logging.info("TF_CONF: {}".format(json.dumps(tf_conf)))

# Cluster setting for cloud
cluster = tf_conf.get("cluster", None)


def inference(x_ph):
    hidden = tf.contrib.layers.fully_connected(x_ph, 10)
    with tf.name_scope("logits"):
        logits = tf.nn.softmax(hidden)
    return logits


def build_loss(y_ph, logits):
    with tf.name_scope("loss"):
        cross_entropy = -tf.reduce_sum(y_ph * tf.log(logits))
    return cross_entropy


def main(_):
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
        with tf.Graph().as_default() as graph:
            with tf.device(device_fn):
                global_step = tf.Variable(0, trainable=False, name="global_step")
                x_ph = tf.placeholder(tf.float32, shape=[None, 784], name="x_ph")
                y_ph = tf.placeholder(tf.float32, shape=[None, 10], name="y_ph")
                logits = inference(x_ph)
                loss = build_loss(y_ph, logits)
                tf.scalar_summary("loss", loss)
                train_op = tf.train.GradientDescentOptimizer(0.002).minimize(
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
                saver = tf.train.Saver()

        sv = tf.train.Supervisor(
            graph=graph,
            is_chief=(tf_conf["task"]["type"] == "master"),
            logdir=args.output_path,
            init_op=init_op,
            global_step=global_step,
            summary_op=None
        )

        with sv.managed_session(server.target) as sess:
            # summary_writer = tf.train.SummaryWriter("{}/test".format(args.output_path), sess.graph)
            # print "/job:{0}/task:{1} waiting".format(args.job_name, args.task_index)
            mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
            for i in range(1000):
                x_batch, y_batch = mnist.train.next_batch(200)
                iter, _ = sess.run([global_step, train_op], feed_dict={x_ph: x_batch, y_ph: y_batch})
                # Write summary if server is master
                if tf_conf["task"]["type"] == "master" and i % 10 == 0:
                    fd = {x_ph: mnist.test.images, y_ph: mnist.test.labels}
                    sv.summary_computed(sess, sess.run(summary_op, feed_dict=fd), global_step=i)
                    # summary_str = sess.run(summary_op, feed_dict=fd)
                    logging.info(
                        "save scalar summary (iter: {0} type: {1} index: {2})".format(
                            i, tf_conf["task"]["type"], tf_conf["task"]["index"]
                        )
                    )
                # summary_writer.add_summary(summary_str, i)
                if iter % 100 == 0:
                    acc = sess.run(accuracy, feed_dict={x_ph: mnist.test.images, y_ph: mnist.test.labels})
                    logging.info("global_step: {}".format(iter))
                    logging.info(
                        "- /job:{0}/task:{1} - Accuracy: {2}".format(
                            tf_conf["task"]["type"], tf_conf["task"]["index"], acc
                        )
                    )
            # Export prediction graph
            if tf_conf["task"]["type"] == "master":
                with tf.Graph().as_default():
                    logging.info("save model to {}".format(args.output_path))
                    x = tf.placeholder(tf.float32, shape=[None, 784], name="x_ph")
                    p = inference(x)
                    # Define key element
                    input_key = tf.placeholder(tf.int64, [None, ], name="key")
                    output_key = tf.identity(input_key)
                    # Define API inputs/outpus object
                    inputs = {"key": input_key.name, "image": x.name}
                    outputs = {"key": output_key.name, "scores": p.name}
                    tf.add_to_collection("inputs", json.dumps(inputs))
                    tf.add_to_collection("outputs", json.dumps(outputs))
                    # Save model
                    tf.train.Saver().export_meta_graph(filename="{}/model/export.meta".format(args.output_path))
                    saver.save(sess, "{}/model/export".format(args.output_path), write_meta_graph=False)
            sv.stop()
            # if tf_conf["task"]["type"] == "master":
            #     sv.request_stop()
            # else:
            #     sv.stop()

if __name__ == '__main__':
    tf.app.run()
