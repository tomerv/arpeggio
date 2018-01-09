import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d import Axes3D
from . import Plot

class SampleSet:
    def __init__(self, x, y):
        assert len(x.shape) == 2, x.shape
        assert len(y.shape) == 1, y.shape
        assert x.shape[0] == y.shape[0], (x.shape, y.shape)
        self.x = np.array(x, copy=True)
        self.y = np.array(y, copy=True)
        self.num_classes = int(np.max(self.y) + 1)
    def shuffle(self):
        permutation = np.random.permutation(self.y.shape[0])
        self.x = self.x[permutation,:]
        self.y = self.y[permutation]
    def get_onehot(self):
        labels = (np.arange(self.num_classes) == self.y[:,None]).astype(np.float32)
        assert labels.shape == (self.x.shape[0],self.num_classes)
        return labels
    def get_dist(self):
        mu = np.mean(self.x, 0)
        assert mu.shape == (self.x.shape[1],)
        sigma = np.cov(self.x, rowvar=False)
        assert sigma.shape == (self.x.shape[1],self.x.shape[1])
        return mu,sigma
    def get_train_val(self, num_train_samples):
        assert num_train_samples < self.x.shape[0], (num_train_samples, self.x.shape)
        x_train = self.x[:num_train_samples, :]
        x_val   = self.x[num_train_samples:, :]
        y_train = self.y[:num_train_samples]
        y_val   = self.y[num_train_samples:]
        cls = self.__class__
        return cls(x_train, y_train), cls(x_val, y_val)
    def get_balanced_train_val(self, num_train_samples_per_class):
        assert num_train_samples_per_class * self.num_classes < self.x.shape[0], \
            (num_train_samples_per_class, self.num_classes, self.x.shape)
        x_train = []
        x_val   = []
        y_train = []
        y_val   = []
        for i in range(self.num_classes):
            samx = self.x[self.y==i, :]
            samy = self.y[self.y==i]
            x_train.append(samx[:num_train_samples_per_class, :])
            x_val.append(  samx[num_train_samples_per_class:, :])
            y_train.append(samy[:num_train_samples_per_class])
            y_val.append(  samy[num_train_samples_per_class:])
        cls = self.__class__
        train = cls(np.concatenate(x_train), np.concatenate(y_train))
        val   = cls(np.concatenate(x_val  ), np.concatenate(y_val  ))
        assert train.x.shape[0] == num_train_samples_per_class * self.num_classes
        assert train.x.shape[0] + val.x.shape[0] == self.x.shape[0]
        return train, val


class SampleSetPlottable(SampleSet):
    def __init__(self, x, y):
        SampleSet.__init__(self, x, y)
    def plot(self):
        axes = self._plot_get_axes() # to be implemented by the subclass
        axes.set_aspect('equal')
        axes.autoscale()
        self._plot_dist(axes)     # to be implemented by the subclass
        self._plot_samples(axes)  # to be implemented by the subclass
        return axes

class SampleSet2d(SampleSetPlottable):
    def __init__(self, x, y):
        assert x.shape == (y.shape[0],2), (x.shape, y.shape)
        SampleSetPlottable.__init__(self, x, y)
    def _plot_get_axes(self):
        # create a new plot
        axes = plt.figure().add_subplot(111)
        return axes
    def _plot_dist(self, axes):
        mu, sigma = self.get_dist()
        axes.scatter(mu[0], mu[1], c='purple', edgecolors='none')
        U, s, V = np.linalg.svd(sigma, full_matrices=True)
        angle = np.rad2deg(np.sign(U[0,1]) * np.arccos(U[0,0])) # is this always correct?
        width = np.sqrt(s[0])
        height = np.sqrt(s[1])
        self._plot_dist_ellipse(axes, mu, width  , height  , angle)
        self._plot_dist_ellipse(axes, mu, width*2, height*2, angle)
        self._plot_dist_ellipse(axes, mu, width*4, height*4, angle)
    def _plot_dist_ellipse(self, axes, xy, width, height, angle):
        ell = Ellipse(xy=xy, width=width, height=height, angle=angle, fill=False, ec='purple', linestyle='--')
        axes.add_patch(ell)
    def _plot_samples(self, axes):
        for i in range(self.num_classes):
            sam = self.x[self.y==i, :]
            axes.scatter(sam[:,0], sam[:,1], c=Plot.get_color(i), edgecolors='none')
        if len(self.x) < 20:
            for i in range(len(self.x)):
                axes.annotate(str(i), self.x[i])

class SampleSet3d(SampleSetPlottable):
    def __init__(self, x, y):
        assert x.shape == (y.shape[0],3), (x.shape, y.shape)
        SampleSetPlottable.__init__(self, x, y)
    def _plot_get_axes(self):
        # create a new plot
        axes = plt.figure().add_subplot(111, projection='3d')
        return axes
    def _plot_dist(self, axes):
        pass
    def _plot_samples(self, axes):
        for i in range(self.num_classes):
            sam = self.x[self.y==i, :]
            axes.scatter(sam[:,0], sam[:,1], sam[:,2], s=6**2, c=Plot.get_color(i), edgecolors='none')

class SampleSetBuilder:
    def __init__(self):
        self.samples_per_class = {}
        self.num_classes = 0
    def add_class(self, samples):
        self.samples_per_class[self.num_classes] = samples
        self.num_classes += 1
    def get_sampleset(self, shuffle=True):
        x = np.concatenate(list(self.samples_per_class.values())).astype(np.float32)
        t = [[k]*v.shape[0] for k,v in self.samples_per_class.items()]
        y = np.concatenate(t).astype(np.float32)
        sampleset = self._get_sampleset_inner(x, y) # implemented by the subclass
        if shuffle:
            sampleset.shuffle()
        return sampleset
    def _get_sampleset_inner(self, x, y): # subclass can override
        return SampleSet(x, y)

class SampleSet2dBuilder(SampleSetBuilder):
    def __init__(self):
        SampleSetBuilder.__init__(self)
    def _get_sampleset_inner(self, x, y):
        return SampleSet2d(x, y)
    def add_class(self, samples):
        assert len(samples.shape) == 2, samples.shape
        assert samples.shape[1] == 2, samples.shape
        SampleSetBuilder.add_class(self, samples)
    def add_normal_class(self, mean, cov, rotation, size):
        s = np.random.multivariate_normal(mean=[0., 0.], cov=cov, size=size)
        s1 = (self._rotation_2d(rotation) * s.T).T + mean
        self.add_class(s1)
    def _rotation_2d(self, theta):
        return np.matrix([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

class SampleSet3dBuilder(SampleSetBuilder):
    def __init__(self):
        SampleSetBuilder.__init__(self)
    def add_class(self, samples):
        assert len(samples.shape) == 2, samples.shape
        assert samples.shape[1] == 3, samples.shape
        SampleSetBuilder.add_class(self, samples)
    def _get_sampleset_inner(self, x, y):
        return SampleSet3d(x, y)

########## Some basic sample sets

def gen_triangle(num_samples_per_class=50):
    builder = SampleSet2dBuilder()
    cov = np.matrix([[9., 0.], [0., 1.]])
    builder.add_normal_class([-3.5,  2.], cov,  np.pi/3, num_samples_per_class)
    builder.add_normal_class([ 3.5,  2.], cov, -np.pi/3, num_samples_per_class)
    builder.add_normal_class([  0., -5.], cov,        0, num_samples_per_class)
    return builder.get_sampleset()

def gen_arc(num_samples_per_class=50):
    builder = SampleSet2dBuilder()
    cov = np.matrix([[9., 0.], [0., 1.]])
    builder.add_normal_class([-16,  6], cov,  np.pi/6, num_samples_per_class)
    builder.add_normal_class([ -6,  6], cov, -np.pi/6, num_samples_per_class)
    builder.add_normal_class([  4,  0], cov,        0, num_samples_per_class)
    builder.add_normal_class([ 12, -6], cov, -np.pi/6, num_samples_per_class)
    return builder.get_sampleset()
