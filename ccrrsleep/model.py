import tensorflow as tf

from ccrrsleep.nn import *
from ccrrsleep.loss import *

from ccrrsleep.cbam1d import *

class CCRRFeatureNet(object):

    def __init__(
            self,
            batch_size,
            input_dims,
            n_classes,
            is_train,
            reuse_params,
            use_dropout,
            name="ccrrfeaturenet"
    ):
        self.batch_size = batch_size
        self.input_dims = input_dims
        self.n_classes = n_classes
        self.is_train = is_train
        self.reuse_params = reuse_params
        self.use_dropout = use_dropout
        self.name = name

        self.activations = []
        self.layer_idx = 1
        self.monitor_vars = []

    def _build_placeholder(self):
        # Input
        name = "x_train" if self.is_train else "x_valid"
        self.input_var = tf.compat.v1.placeholder(
            tf.float32,
            shape=[self.batch_size, self.input_dims, 1, 1],
            name=name + "_inputs"
        )
        # Target
        self.target_var = tf.compat.v1.placeholder(
            tf.int32,
            shape=[self.batch_size, ],
            name=name + "_targets"
        )

    def _bn_activate_layer(self, input_var):
        name = "l{}_bn_activate".format(self.layer_idx)
        with tf.compat.v1.variable_scope(name) as scope:
            output = batch_norm_new(name="bn", input_var=input_var, is_train=self.is_train)
            # output = tf.nn.relu(output, name="relu")
            # output = tf.nn.swish(output, name="swish")
            output = mish(output)
        self.activations.append((name, output))
        self.layer_idx += 1
        return output

    def _conv1d_layer(self, input_var, filter_size, n_filters, stride=1, dilations=1, padding="SAME", wd=0.0):
        input_shape = input_var.get_shape()
        n_batches = input_shape[0].value
        input_dims = input_shape[1].value
        n_in_filters = input_shape[3].value
        name = "l{}_conv".format(self.layer_idx)
        with tf.compat.v1.variable_scope(name) as scope:
            output = conv_1d(name="conv1d", input_var=input_var, filter_shape=[filter_size, 1, n_in_filters, n_filters],
                             stride=stride, padding=padding, dilations=dilations, bias=None, wd=wd)
            output = self._bn_activate_layer(output)
        self.activations.append((name, output))
        self.layer_idx += 1
        return output

    def _mcb_layer(self, input_var, n_filters=64):
        name = "l{}_mcb_conv".format(self.layer_idx)
        with tf.compat.v1.variable_scope(name) as scope:
            output_x = self._conv1d_layer(input_var=input_var, filter_size=1, n_filters=n_filters, dilations=1)

        name = "l{}_res_conv1".format(self.layer_idx)
        with tf.compat.v1.variable_scope(name) as scope:
            output = self._conv1d_layer(input_var=input_var, filter_size=1, n_filters=n_filters, dilations=1)
            output_res1 = self._conv1d_layer(input_var=output, filter_size=3, n_filters=n_filters, dilations=1)

        name = "l{}_res_conv2".format(self.layer_idx)
        with tf.compat.v1.variable_scope(name) as scope:
            output = self._conv1d_layer(input_var=input_var, filter_size=1, n_filters=n_filters, dilations=1)
            output = self._conv1d_layer(input_var=output, filter_size=3, n_filters=n_filters, dilations=1)
            output_res2 = self._conv1d_layer(input_var=output, filter_size=3, n_filters=n_filters, dilations=3)

        name = "l{}_res_conv3".format(self.layer_idx)
        with tf.compat.v1.variable_scope(name) as scope:
            output = self._conv1d_layer(input_var=input_var, filter_size=1, n_filters=n_filters, dilations=1)
            output = self._conv1d_layer(input_var=output, filter_size=3, n_filters=n_filters, dilations=1)
            output = self._conv1d_layer(input_var=output, filter_size=3, n_filters=n_filters, dilations=1)
            output_res3 = self._conv1d_layer(input_var=output, filter_size=3, n_filters=n_filters, dilations=5)

        output = self._concat_layer([output_x, output_res1, output_res2, output_res3], axis=-1)
        output = self._bn_activate_layer(output)
        self.activations.append((name, output))
        self.layer_idx += 1
        return output

    def _max_pool_1d_layer(self, input_var, pool_size=2,stride=None):
        name = "l{}_pool".format(self.layer_idx)
        if not stride:
            stride = pool_size
        output = max_pool_1d(name=name, input_var=input_var, pool_size=pool_size, stride=stride)
        self.activations.append((name, output))
        self.layer_idx += 1
        return output

    def _avg_pool_1d_layer(self, input_var, pool_size=2, stride=None):
        name = "l{}_pool".format(self.layer_idx)
        if not stride:
            stride = pool_size
        output = avg_pool_1d(name=name, input_var=input_var, pool_size=pool_size, stride=stride)
        self.activations.append((name, output))
        self.layer_idx += 1
        return output

    def _global_avg_pool_1d_layer(self, input_var):
        input_shape = input_var.get_shape()
        pool_size = input_shape[1].value
        name = "l{}_global_avg_pool_1d".format(self.layer_idx)
        output = avg_pool_1d(name=name, input_var=input_var, pool_size=pool_size, stride=pool_size)
        self.activations.append((name, output))
        self.layer_idx += 1
        return output

    def _dropout_layer(self, input_var, keep_prob=0.5):
        if self.use_dropout:
            name = "l{}_dropout".format(self.layer_idx)
            with tf.compat.v1.variable_scope(name) as scope:
                if self.is_train:
                    output = tf.nn.dropout(input_var, keep_prob=keep_prob, name=name)
                else:
                    output = tf.nn.dropout(input_var, keep_prob=1.0, name=name)
                self.activations.append((name, output))
        else:
            output = input_var
        self.layer_idx += 1
        return output

    def _attention_layer(self, input_var, mode='cbam', ratio=4, kernel_size=15):
        name = "l{}_attention".format(self.layer_idx)
        with tf.compat.v1.variable_scope(name) as scope:
            if mode =='channel_attention':
                output = channel_attention(input_var, name=name, ratio=ratio)
            elif mode =='spatial_attention':
                output = spatial_attention(input_var, name=name, kernel_size=kernel_size)
            else:
                output = cbam_block(input_var, name=name, ratio=4)
            # output = se_block(input_var, name=name,ratio=4)
            output = tf.add(input_var,output)
        self.activations.append((name, output))
        self.layer_idx += 1
        return output

    def _flatten_layer(self, input_var):
        name = "l{}_flat".format(self.layer_idx)
        output = flatten(name=name, input_var=input_var)
        self.activations.append((name, output))
        self.layer_idx += 1
        return output

    def _concat_layer(self, values, axis=1):
        name = "l{}_concat".format(self.layer_idx)
        output = tf.concat(axis=axis, values=values, name=name)
        self.activations.append((name, output))
        self.layer_idx += 1
        return output

    def _add_layer(self, values):
        name = "l{}_add".format(self.layer_idx)
        output = tf.add_n(values, name=name)
        self.activations.append((name, output))
        self.layer_idx += 1
        return output

    def _fc_layer(self, input_var, n_hiddens, bias=0.0, wd=0.):
        name = "l{}_linear".format(self.layer_idx)
        output = fc(name=name, input_var=input_var, n_hiddens=n_hiddens, bias=bias, wd=wd)
        # output = self._bn_activate_layer(output)
        self.activations.append((name, output))
        self.layer_idx += 1
        return output

    def _fragment_gru(self, input_var, gru_layers=1, hidden_size=512, use_dropout=False,use_attention=False):
        name = "l{}_bi_gru".format(self.layer_idx)
        with tf.compat.v1.variable_scope(name) as scope:
            def gru_cell():
                cell = tf.compat.v1.nn.rnn_cell.GRUCell(hidden_size)
                if use_dropout:
                    keep_prob = 0.5 if self.is_train else 1.0
                    cell = tf.compat.v1.nn.rnn_cell.DropoutWrapper(
                        cell,
                        output_keep_prob=keep_prob
                    )
                return cell

            fw_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell([gru_cell() for _ in range(gru_layers)],
                                                            state_is_tuple=True)
            bw_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell([gru_cell() for _ in range(gru_layers)],
                                                            state_is_tuple=True)
            # Feedforward to MultiRNNCell
            list_rnn_inputs = tf.unstack(input_var, axis=1)
            # outputs, fw_state, bw_state = tf.nn.bidirectional_rnn(
            outputs, fw_state, bw_state = tf.compat.v1.nn.static_bidirectional_rnn(
                cell_fw=fw_cell,
                cell_bw=bw_cell,
                inputs=list_rnn_inputs,
                dtype=tf.float32
            )
            if use_attention:
                T = len(outputs)
                network = tf.reshape(tf.concat(axis=1, values=outputs), [-1, hidden_size * 2,1,T])
                network = self._attention_layer(network, mode='channel_attention')
                network = tf.reduce_mean(network,axis=-1, keepdims=False)
            else:
                network = outputs[-1]
            self.activations.append((name, network))
            self.layer_idx += 1
        return network

    def _fragment_cnn(self, input_var):
        network_25 = self._conv1d_layer(input_var=input_var, filter_size=25, n_filters=64, stride=6, wd=1e-3)
        network_25 = self._max_pool_1d_layer(network_25, pool_size=5)

        network_100 = self._conv1d_layer(input_var=input_var, filter_size=100, n_filters=64, stride=15, wd=1e-3)
        network_100 = self._max_pool_1d_layer(network_100, pool_size=2)

        network = self._concat_layer([network_25, network_100], axis=-1)
        # network = self._attention_layer(network, mode='cbam')
        # # # Dropout
        network = self._dropout_layer(network, keep_prob=0.5)

        # 300
        # Convolution
        network = self._mcb_layer(network, n_filters=64)
        print(network.shape)
        # ssee.run(network)
        # plt.plot(network)
        # network = self._dropout_layer(network)
        # network = self._attention_layer(network, mode='cbam')
        # --------------------------------------
        # network_res = self._global_avg_pool_1d_layer(network)
        # network_res = self._conv1d_layer(input_var=network_res, filter_size=1, n_filters=1024, stride=1)
        network_res = self._conv1d_layer(input_var=network, filter_size=1, n_filters=1024, stride=1)
        network_res = self._global_avg_pool_1d_layer(network_res)
        # # network_res = self._dropout_layer(network_res)
        network_res = self._flatten_layer(network_res)
        # --------------------------------------
        network_cnn = self._conv1d_layer(input_var=network, filter_size=1, n_filters=256, stride=1)
        network_cnn = self._max_pool_1d_layer(network_cnn, 10)
        # network_cnn = self._conv1d_layer(input_var=network_cnn, filter_size=3, n_filters=256, stride=1)
        # network_cnn = self._max_pool_1d_layer(network_cnn, 2)
        # network_cnn = self._mcb_layer(network_cnn, n_filters=64)
        # network_cnn = self._max_pool_1d_layer(network_cnn, 3)
        network_cnn = self._conv1d_layer(input_var=network_cnn, filter_size=3, n_filters=1024, stride=1)
        # network_cnn = self._attention_layer(network_cnn, mode ='channel_attention')
        network_cnn = self._global_avg_pool_1d_layer(network_cnn)
        network_cnn = self._flatten_layer(network_cnn)
        # --------------------------------------
        network_rnn = self._conv1d_layer(input_var=network, filter_size=1, n_filters=256, stride=1)
        network_rnn = self._avg_pool_1d_layer(network_rnn, 10)
        network_rnn_shape = network_rnn.get_shape()
        network_rnn_T = network_rnn_shape[1].value
        network_rnn_D = network_rnn_shape[3].value
        network_rnn = tf.reshape(network_rnn, [-1, network_rnn_T, network_rnn_D])
        network_rnn = self._fragment_gru(network_rnn)
        network_rnn = self._flatten_layer(network_rnn)

        network = self._concat_layer([network_cnn, network_rnn, network_res], axis=-1)
        # network = self._concat_layer([network_cnn, network_rnn], axis=-1)
        # network = self._add_layer([network,network_res])
        # network = self._dropout_layer(network,keep_prob=0.5)
        return network

    def build_model(self, input_var):
        network = self._fragment_cnn(input_var)
        return network

    def init_ops(self):
        self._build_placeholder()

        # Get loss and prediction operations
        with tf.compat.v1.variable_scope(self.name) as scope:
            # Reuse variables for validation
            if self.reuse_params:
                scope.reuse_variables()

            # Build model
            network = self.build_model(input_var=self.input_var)

            # network = self._fc_layer(network, self.n_classes)
            # Softmax linear
            name = "l{}_softmax_linear".format(self.layer_idx)
            network = fc(name=name, input_var=network, n_hiddens=self.n_classes, bias=0.0, wd=0)
            self.activations.append((name, network))
            self.layer_idx += 1

            # Outputs of softmax linear are logits
            self.logits = network

            ######### Compute loss #########
            # Cross-entropy loss
            # loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            #     logits=self.logits,
            #     labels=self.target_var,
            #     name="sparse_softmax_cross_entropy_with_logits"
            # )
            loss = focal_loss(self.target_var, self.logits)
            # loss = ghmc_loss(self.target_var, self.logits)
            loss = tf.reduce_mean(loss, name="cross_entropy")

            # Regularization loss
            regular_loss = tf.add_n(
                tf.compat.v1.get_collection("losses", scope=scope.name + "\/"),
                name="regular_loss"
            )

            # Total loss
            self.loss_op = tf.add(loss, regular_loss)

            # Predictions
            self.pred_op = tf.argmax(self.logits, 1)


class CCRRSleepNet(CCRRFeatureNet):

    def __init__(
            self,
            batch_size,
            input_dims,
            n_classes,
            seq_length,
            n_rnn_layers,
            return_last,
            is_train,
            reuse_params,
            use_dropout_feature,
            use_dropout_sequence,
            name="ccrrsleepnet"
    ):
        super(self.__class__, self).__init__(
            batch_size=batch_size,
            input_dims=input_dims,
            n_classes=n_classes,
            is_train=is_train,
            reuse_params=reuse_params,
            use_dropout=use_dropout_feature,
            name=name
        )

        self.seq_length = seq_length
        self.n_rnn_layers = n_rnn_layers
        self.return_last = return_last

        self.use_dropout_sequence = use_dropout_sequence

    def _build_placeholder(self):
        # Input
        name = "x_train" if self.is_train else "x_valid"
        self.input_var = tf.compat.v1.placeholder(
            tf.float32,
            shape=[self.batch_size * self.seq_length, self.input_dims, 1, 1],
            name=name + "_inputs"
        )
        # Target
        self.target_var = tf.compat.v1.placeholder(
            tf.int32,
            shape=[self.batch_size * self.seq_length, ],
            name=name + "_targets"
        )

    def build_model(self, input_var):
        # Create a network with superclass method
        network = super(self.__class__, self).build_model(
            input_var=self.input_var
        )

        # Residual (or shortcut) connection
        output_conns = []

        # Fully-connected to select some part of the output to add with the output from bi-directional LSTM
        name = "l{}_fc".format(self.layer_idx)
        with tf.compat.v1.variable_scope(name) as scope:
            output = fc(name="fc", input_var=network, n_hiddens=1024, bias=None, wd=0)
            output = batch_norm_new(name="bn", input_var=output, is_train=self.is_train)
            output = mish(output)
        self.activations.append((name, output))
        self.layer_idx += 1
        output_conns.append(output)

        # output_conns.append(network)

        ######################################################################

        # Reshape the input from (batch_size * seq_length, input_dim) to
        # (batch_size, seq_length, input_dim)
        name = "l{}_reshape_seq".format(self.layer_idx)
        input_dim = network.get_shape()[-1].value
        seq_input = tf.reshape(network,
                               shape=[-1, self.seq_length, input_dim],
                               name=name)
        assert self.batch_size == seq_input.get_shape()[0].value
        self.activations.append((name, seq_input))
        self.layer_idx += 1

        # Bidirectional LSTM network
        name = "l{}_bi_lstm".format(self.layer_idx)
        hidden_size = 512  # will output 1024 (512 forward, 512 backward)
        with tf.compat.v1.variable_scope(name) as scope:

            def lstm_cell():

                cell = tf.compat.v1.nn.rnn_cell.LSTMCell(hidden_size,
                                                         use_peepholes=True,
                                                         state_is_tuple=True,
                                                         reuse=tf.compat.v1.get_variable_scope().reuse)
                if self.use_dropout_sequence:
                    keep_prob = 0.5 if self.is_train else 1.0
                    cell = tf.compat.v1.nn.rnn_cell.DropoutWrapper(
                        cell,
                        output_keep_prob=keep_prob
                    )

                return cell

            fw_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell([lstm_cell() for _ in range(self.n_rnn_layers)],
                                                            state_is_tuple=True)
            bw_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell([lstm_cell() for _ in range(self.n_rnn_layers)],
                                                            state_is_tuple=True)

            # Initial state of RNN
            self.fw_initial_state = fw_cell.zero_state(self.batch_size, tf.float32)
            self.bw_initial_state = bw_cell.zero_state(self.batch_size, tf.float32)

            # Feedforward to MultiRNNCell
            list_rnn_inputs = tf.unstack(seq_input, axis=1)
            # outputs, fw_state, bw_state = tf.nn.bidirectional_rnn(
            outputs, fw_state, bw_state = tf.compat.v1.nn.static_bidirectional_rnn(
                cell_fw=fw_cell,
                cell_bw=bw_cell,
                inputs=list_rnn_inputs,
                initial_state_fw=self.fw_initial_state,
                initial_state_bw=self.bw_initial_state
            )

            if self.return_last:
                network = outputs[-1]
            else:
                network = tf.reshape(tf.concat(axis=1, values=outputs), [-1, hidden_size * 2],
                                     name=name)
            self.activations.append((name, network))
            self.layer_idx += 1

            self.fw_final_state = fw_state
            self.bw_final_state = bw_state

        # Append output
        output_conns.append(network)

        ######################################################################

        # Concat
        # network = tf.concat(output_conns, axis=1)

        # Add
        name = "l{}_add".format(self.layer_idx)
        network = tf.add_n(output_conns, name=name)
        self.activations.append((name, network))
        self.layer_idx += 1

        # Dropout
        if self.use_dropout_sequence:
            name = "l{}_dropout".format(self.layer_idx)
            if self.is_train:
                network = tf.nn.dropout(network, keep_prob=0.5, name=name)
            else:
                network = tf.nn.dropout(network, keep_prob=1.0, name=name)
            self.activations.append((name, network))
        self.layer_idx += 1

        return network

    def init_ops(self):
        self._build_placeholder()

        # Get loss and prediction operations
        with tf.compat.v1.variable_scope(self.name) as scope:
            # Reuse variables for validation
            if self.reuse_params:
                scope.reuse_variables()

            # Build model
            network = self.build_model(input_var=self.input_var)

            # Softmax linear
            name = "l{}_softmax_linear".format(self.layer_idx)
            network = fc(name=name, input_var=network, n_hiddens=self.n_classes, bias=0.0, wd=0)
            self.activations.append((name, network))
            self.layer_idx += 1

            # Outputs of softmax linear are logits
            self.logits = network
            # Weighted cross-entropy loss for a sequence of logits (per example)
            loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
                [self.logits],
                [self.target_var],
                [tf.ones([self.batch_size * self.seq_length])],
                softmax_loss_function=focal_loss,
                # softmax_loss_function=ghmc_loss,
                name="sequence_loss_by_example"
            )
            loss = tf.reduce_sum(loss) / self.batch_size

            # Regularization loss
            regular_loss = tf.add_n(
                tf.compat.v1.get_collection("losses", scope=scope.name + "\/"),
                name="regular_loss"
            )

            # Total loss
            self.loss_op = tf.add(loss, regular_loss)

            # Predictions
            self.pred_op = tf.argmax(self.logits, 1)
