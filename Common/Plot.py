import numpy as np

def get_color(idx):
    colors = ['red', 'green', 'blue', 'darkorange', 'magenta', 'brown']
    res = colors[idx] if idx < len(colors) else 'gray'
    return res

def get_plot_bounding_box(all_samples, margin):
    xmin = np.min(all_samples[:,0])
    xmax = np.max(all_samples[:,0])
    xscale = xmax - xmin
    ymin = np.min(all_samples[:,1])
    ymax = np.max(all_samples[:,1])
    yscale = ymax - ymin
    scale = (xscale+yscale)/2 + 2*margin
    return (xmin - xscale*margin, xmax + xscale*margin, ymin - yscale*margin, ymax + yscale*margin, scale)

def plot_hyperplanes(axes, all_samples, weights, biases):
    xmin, xmax, ymin, ymax, scale = get_plot_bounding_box(all_samples, 0.2)
    for i in range(weights.shape[1]):
        w0, w1, b = weights[0,i], weights[1,i], biases[0,i]
        # plot the line w0*x+w1*y+b=0
        # find start and end points, by intersecting the line with the bounding box
        intersection_with_y = lambda y: -(b + w1*y) / w0
        p = [xmin, xmax, intersection_with_y(ymin), intersection_with_y(ymax)]
        xstart,xend = sorted(p)[1:3]
        if xstart < xmin or xend > xmax:
            # line is outside of bounding box
            continue
        # plot the segment from xstart to xend
        ystart = (-w0 * xstart - b) / w1
        yend   = (-w0 * xend - b) / w1
        # plot 2 parallel lines
        dist = scale / 50
        xdelta = dist * w0 / np.sqrt(np.square(w0) + np.square(w1))
        ydelta = dist * w1 / np.sqrt(np.square(w0) + np.square(w1))
        axes.plot([xstart+xdelta, xend+xdelta], [ystart+ydelta, yend+ydelta],  '-', c=get_color(i))
        axes.plot([xstart-xdelta, xend-xdelta], [ystart-ydelta, yend-ydelta], '--', c=get_color(i))

