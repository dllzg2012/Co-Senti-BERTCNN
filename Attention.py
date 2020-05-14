from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import  Layer
from tensorflow.python.keras import initializers, regularizers, constraints


class Test_IAN(Layer):
    def __init__(self, step_dim,ll,get_alpha=False,get_sequence=False,
                 W_regularizer=None, b_regularizer=None,L_regularizer=None,
                 W_constraint=None, b_constraint=None,L_constraint=None,
                 bias=False,  **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')
        self.l_init = initializers.constant(value=0.5)
        self.ll=ll
        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.L_regularizer = regularizers.get(L_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)
        self.L_constraint = constraints.get(L_constraint)
        self.get_sequence=get_sequence
        self.bias=bias
        self.step_dim = step_dim
        self.features_dim = 0
        self.get_alpha=get_alpha
        super(Test_IAN, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape[0]) == 3

        self.W1 = self.add_weight(shape=(int(input_shape[0][-1] * input_shape[0][-1]),),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.W2 = self.add_weight(shape=(int(input_shape[0][-1] * input_shape[0][-1]),),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)

        self.features_dim = input_shape[0][-1]
        if self.bias:
            self.b = self.add_weight(shape=(int(input_shape[0][1]),),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim
        #xw = K.reshape(K.dot(x[0], K.reshape(self.W, (features_dim, features_dim))), (-1, features_dim))
        #yavg=K.reshape(K.mean(K.mean(x[1], axis=1, keepdims=True),axis=0, keepdims=True), (features_dim,-1))
        xw1=K.dot(x[0], K.reshape(self.W1, (features_dim, features_dim)))
        xw2 = K.dot(x[1], K.reshape(self.W2, (features_dim, features_dim)))
        xw1t=K.permute_dimensions(xw1,[0,2,1])
        xw2t = K.permute_dimensions(xw2, [0, 2, 1])
        xw11=K.batch_dot(xw1,xw1t)/ (step_dim ** 0.5)
        xw12 = K.batch_dot(xw1, xw2t)/ (step_dim ** 0.5)

        s11=self.ll*K.softmax(xw11)
        s12=(1-self.ll)*K.softmax(xw12)

        eij=s11+s12
        print(eij.get_shape())
        V=x[0]*K.mean(eij,axis=2,keepdims=True)
        if self.get_alpha:
            return eij
        else:
            if self.get_sequence:
                return V
            else:
                return K.sum(V, axis=1)

    def compute_output_shape(self, input_shape):
        #return input_shape[0][0], input_shape[0][1],input_shape[0][2]
        if self.get_alpha:
            return input_shape[0][0],input_shape[0][1],input_shape[0][1]
        else:
            if self.get_sequence:
                return input_shape[0][0], input_shape[0][1], self.features_dim
            else:
                return input_shape[0][0], self.features_dim