import tensorflow as tf
from utils.numtoWord import createNum2WordDict, vocab


def generator_function(params):
    while True:
        d = createNum2WordDict(size=100, high=params['max_num'])
        for value, word in d.items():
            if value == 0:
                continue
            ids = [vocab[char] for char in word]
            length = len(word)
            yield (ids, length, value)


def input_fn(params):
    generator = lambda: generator_function(params)
    dataset = tf.data.Dataset.from_generator(
        generator=generator,
        output_types=(tf.int64, tf.int64, tf.double),
        output_shapes=(tf.TensorShape([None]), tf.TensorShape([]), tf.TensorShape([]))
    )
    dataset = dataset.padded_batch(
        params['batch_size'],
        padded_shapes=(tf.TensorShape([None]), tf.TensorShape([]), tf.TensorShape([]))
    )

    dataset = dataset.map(lambda x, y, z: ({"sequences": x, "lengths": y}, z %2))
    return dataset

def model_fn_factory(model,crossEntropy=False):
    '''

    :param model: A python function implementing a tensorflow model which receives features and returns logits
    :return:
    '''
    def model_fn(features,labels,mode,params,):
        with tf.variable_scope("model"):
            logits = model(features,labels,params)
            if crossEntropy ==True:
                loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=labels)
            else:
                loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,labels=labels)
            loss = tf.reduce_mean(loss)
            tf.summary.scalar("loss",loss)
            opt = tf.train.AdamOptimizer(0.0001)
            train = opt.minimize(loss,global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train)
    return model_fn

def estimator_factory(model,params,crossEntropy=False):
    model_fn = model_fn_factory(model,crossEntropy)
    estimator = tf.estimator.Estimator(model_fn=model_fn, params=params)
    return estimator