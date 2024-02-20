from imports import *
def plot(self, h=0.01):
    """
    Generate plot of input data and decision boundary.
    """
    # setting plot properties like size, theme and axis limits
    sns.set_style('darkgrid')
    plt.figure(figsize=(20, 20))

    plt.axis('scaled')
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)

    colors = {
        0: "ro",
        1: "go"
    }

    # plotting the four datapoints
    for i in range(len(self.train_data)):
        plt.plot([self.train_data[i][0]],
                 [self.train_data[i][1]],
                 colors[self.target[i][0]],
                 markersize=20)

    x_range = np.arange(-0.1, 1.1, h)
    y_range = np.arange(-0.1, 1.1, h)

    # creating a mesh to plot decision boundary
    xx, yy = np.meshgrid(x_range, y_range, indexing='ij')
    Z = np.array([[self.classify([x, y]) for x in x_range] for y in y_range])

    # using the contourf function to create the plot
    plt.contourf(xx, yy, Z, colors=['red', 'green', 'green', 'blue'], alpha=0.4)