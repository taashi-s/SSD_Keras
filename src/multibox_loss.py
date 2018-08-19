import keras.backend as KB
import tensorflow as tf


class MultiboxLoss(object):
    def __init__(self, num_classes, alpha=1.0, negative_ratio=2.0
                 , negatives_for_hard=100.0):
        self.__num_classes = num_classes
        self.__alpha = alpha
        self.__negative_ratio = negative_ratio
        self.__negatives_for_hard = negatives_for_hard


    def loss(self, y_true, y_pred):
        batch_size = KB.shape(y_true)[0]
        box_num = KB.cast(KB.shape(y_true)[1], 'float32')

        loc_loss = self.__smooth_loss(y_true[:, :, :4], y_pred[:, :, :4])
        conf_loss = self.__softmax_loss(y_true[:, :, 4:-8], y_pred[:, :, 4:-8])

        positive_num = tf.reduce_sum(y_true[:, :, -8], axis=-1)
        positive_loc_loss = tf.reduce_sum(loc_loss * y_true[:, :, -8], axis=1)
        positive_conf_loss = tf.reduce_sum(conf_loss * y_true[:, :, -8], axis=1)

        negative_num_batch = self.__calc_negative_num_batch(box_num, positive_num)
        negative_conf_loss = self.__get_negative_conf_loss(y_true, y_pred, conf_loss
                                                           , batch_size, box_num, positive_num)

        total_loss = self.__get_total_loss(positive_num, positive_loc_loss, positive_conf_loss
                                           , negative_num_batch, negative_conf_loss)
        return total_loss


    def __smooth_loss(self, y_true, y_pred):
        abs_loss = tf.abs(y_true - y_pred)
        sq_loss = 0.5 * (y_true - y_pred)**2
        l1_loss = tf.where(tf.less(abs_loss, 1.0), sq_loss, abs_loss - 0.5)
        return tf.reduce_sum(l1_loss, -1)


    def __softmax_loss(self, y_true, y_pred):
        y_pred = tf.maximum(tf.minimum(y_pred, 1 - 1e-15), 1e-15)
        softmax_loss = tf.reduce_sum(y_true * tf.log(y_pred), axis=-1)
        return -1 * softmax_loss


    def __calc_negative_num_batch(self, box_num, positive_num):
        negative_num = self.__negative_ratio * positive_num
        negative_num = tf.minimum(negative_num, box_num - positive_num)

        masks = tf.greater(negative_num, 0)
        has_min = tf.to_float(tf.reduce_any(masks))
        values = [negative_num, [(1 - has_min) * self.__negatives_for_hard]]
        negative_num = tf.concat(axis=0, values=values)
        boolean_mask = tf.boolean_mask(negative_num, tf.greater(negative_num, 0))

        negative_num_batch = tf.reduce_min(boolean_mask)
        negative_num_batch = KB.cast((negative_num_batch), 'int32')
        return negative_num_batch


    def __get_negative_conf_loss(self, y_true, y_pred, conf_loss
                                 , batch_size, box_num, negative_num_batch):
        max_confs = tf.reduce_max(y_pred[:, :, 5:self.__num_classes + 4], axis=2)
        _, ids = tf.nn.top_k(max_confs * (1 - y_true[:, :, -8]), k=negative_num_batch)
        batch_ids = tf.expand_dims(tf.range(0, batch_size), 1)
        batch_ids = tf.tile(batch_ids, (1, negative_num_batch))
        full_ids = tf.reshape(batch_ids, [-1]) * tf.to_int32(box_num) + tf.reshape(ids, [-1])

        negative_conf_loss = tf.gather(tf.reshape(conf_loss, [-1]), full_ids)
        negative_conf_loss = tf.reshape(negative_conf_loss, [batch_size, negative_num_batch])
        negative_conf_loss = tf.reduce_sum(negative_conf_loss, axis=1)
        return negative_conf_loss


    def __get_total_loss(self, positive_num, positive_loc_loss, positive_conf_loss
                         , negative_num_batch, negative_conf_loss):
        total_loss = positive_conf_loss + negative_conf_loss
        total_loss /= (positive_num + tf.to_float(negative_num_batch))
        positive_num = tf.where(tf.not_equal(positive_num, 0), positive_num, tf.ones_like(positive_num))
        total_loss += (self.__alpha * positive_loc_loss) / positive_num
        return total_loss
