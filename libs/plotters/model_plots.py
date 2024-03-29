import time
import matplotlib.pyplot as plt
import numpy as np
import libs.model_helpers.linalg as linalg


class CostRT:
    def __init__(self):
        self.costs = []
        self.accuracies = []
        self.fig = None
        self.ax = None
        self.line1 = None
        self.line2 = None
        self.ax1 = None
        self.ax2 = None
        self.timestamps = []
        self.start = None

    def plot(self):
        plt.ion()
        self.fig, self.ax1 = plt.subplots()
        self.ax2 = self.ax1.twinx()

        self.line1 = self.ax1.plot([], [], 'r-')  # Returns a tuple of line objects, thus the comma
        self.line2 = self.ax2.plot([], [], 'g-')  # Returns a tuple of line objects, thus the comma

        self.ax1.set_xlabel('Time (seconds)')

        self.ax1.set_ylabel('Cost (absolute)', color='r')

        self.ax2.set_ylabel('Accuracy (absolute)', color='g')
        self.ax2.set_ylim([0, 1])

        self.start = time.time()

    def add(self, x, x2):
        self.costs.append(x)
        self.accuracies.append(x2)
        self.timestamps.append(time.time() - self.start)

        maxc = self.costs[0]
        for cost in self.costs:
            maxc = max(cost, maxc)

        self.ax1.set_ylim([0, maxc])

    def show(self):

        self.ax1.plot(self.timestamps, self.costs, 'r-')
        self.ax2.plot(self.timestamps, self.accuracies, 'g-')

        self.ax1.relim()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        # plt.ylabel('Cost (absolute)')
        # plt.xlabel('Time (seconds)')
        # plt.legend(loc="best")
        plt.show()


class Train:
    def show(self, x_train, y_train):
        fig, axes = plt.subplots(3, 3, figsize=(5, 5))  # Create a grid of 3x3 sub-plots

        # Flatten the grid of axes to easily iterate over them
        axes = axes.ravel()

        for i in range(9):  # We want to display 9 images
            axes[i].imshow(x_train[i], cmap='gray')  # Display the i-th image in grayscale
            axes[i].set_title(f"Digit: {y_train[i]}")  # Set the title of the plot to the label of the image
            axes[i].axis('off')  # Hide axis details for clarity

        plt.subplots_adjust(wspace=0.5)  # Add some space between the sub-plots
        plt.show()  # Show the plot


class Eval:
    def show_top_cost(self, examples):
        # Sort the examples by cost in descending order
        examples.sort(key=lambda x: x['cost'], reverse=True)

        # Get the top 5 examples
        top_5 = examples[:5]

        # Prepare the subplots
        fig, axes = plt.subplots(1, 5, figsize=(15, 3))
        fig.suptitle("Top costs")

        for i, example in enumerate(top_5):
            # Reshape the flattened image data for display, assuming the original shape was 28x28
            image = example['x'].reshape((28, 28))

            # Display the image
            axes[i].imshow(image, cmap='gray')

            # Set the title with cost, prediction, and correct label
            title = f"Cost: {example['cost']:.2f}\nPredict: {example['prediction']}\nLabel: {example['label']}"
            axes[i].set_title(title)

            # Hide the axes for clarity
            axes[i].axis('off')

        plt.tight_layout()
        plt.show()

    def show_low_correct(self, examples):
        # Sort the examples by cost in descending order
        examples.sort(key=lambda x: x['cost'])

        # Get the top 5 examples
        dict = {}
        i = 0
        while len(dict) < 5:
            label = examples[i]['label']
            dict[label] = examples[i]
            i += 1

        low_5 = []
        for val in dict.values():
            low_5.append(val)

        # Prepare the subplots
        fig, axes = plt.subplots(1, 5, figsize=(15, 3))
        fig.suptitle("Low costs - Correct")

        for i, example in enumerate(low_5):
            # Reshape the flattened image data for display, assuming the original shape was 28x28
            image = example['x'].reshape((28, 28))

            # Display the image
            axes[i].imshow(image, cmap='gray')

            # Set the title with cost, prediction, and correct label
            title = f"Cost: {example['cost']:.2f}\nPredict: {example['prediction']}\nLabel: {example['label']}"
            axes[i].set_title(title)

            # Hide the axes for clarity
            axes[i].axis('off')

        plt.tight_layout()
        plt.show()

    def show_low_incorrect(self, examples):
        # Sort the examples by cost in descending order
        examples.sort(key=lambda x: x['cost'])

        # Get the top 5 examples
        low_5 = [example for example in examples if example['prediction'] != example['label']][:5]

        # Prepare the subplots
        fig, axes = plt.subplots(1, 5, figsize=(15, 3))
        fig.suptitle("Low costs - Incorrect")

        for i, example in enumerate(low_5):
            # Reshape the flattened image data for display, assuming the original shape was 28x28
            image = example['x'].reshape((28, 28))

            # Display the image
            axes[i].imshow(image, cmap='gray')

            # Set the title with cost, prediction, and correct label
            title = f"Cost: {example['cost']:.2f}\nPredict: {example['prediction']}\nLabel: {example['label']}"
            axes[i].set_title(title)

            # Hide the axes for clarity
            axes[i].axis('off')

        plt.tight_layout()
        plt.show()


class ActivationLog:
    def __init__(self):
        self.activations = []
        self.fig = None
        self.ax = None
        self.line1 = None
        self.line2 = None
        self.ax1 = None
        self.ax2 = None

    def plot(self):
        # Create a line plot
        plt.plot(np.arange(len(self.activations) - 1), [np.mean(layer) for layer in self.activations[1:]])

        # Add labels and title
        plt.xlabel('Layer')
        plt.ylabel('Mean activation')
        plt.title('Mean activation')

        # Display the plot
        plt.show()

        # Stdev
        plt.plot(np.arange(len(self.activations) - 1), [np.std(layer) for layer in self.activations[1:]])

        # Add labels and title
        plt.xlabel('Layer')
        plt.ylabel('Stdev activation')
        plt.title('Stdev activation')

        # Display the plot
        plt.show()

        # return

        for i, layer in enumerate(self.activations):
            plt.hist(layer.flatten())
            plt.xlabel('Activation')
            plt.ylabel('Count')
            plt.title(f"Layer {i}")
            plt.show()

    def add(self, x: list[np.ndarray]):
        self.activations = x.copy()

    def show(self):
        # self.fig.canvas.draw()
        # self.fig.canvas.flush_events()
        plt.ylabel('Mean sum')
        plt.xlabel('Layer')
        plt.show()


class WeightGradLog:
    def __init__(self):
        self.w_grad = []
        self.fig = None
        self.ax = None
        self.line1 = None
        self.line2 = None
        self.ax1 = None
        self.ax2 = None

    def plot(self):

        # Create a line plot
        print(len(self.w_grad))
        plt.plot(range(len(self.w_grad)), self.w_grad)

        # Add labels and title
        plt.xlabel('Layer')
        plt.ylabel('Mean w grad')
        plt.title('Mean w grad')

        # Display the plot
        plt.show()

    def add(self, x: list[np.ndarray]):
        x = x.copy()
        x.reverse()
        for layer in x:
            gt_zero = abs(layer[abs(layer) > 10 ** -3])
            if (len(gt_zero) > 0):
                self.w_grad.append(gt_zero.mean())
            else:
                self.w_grad.append(0)
        # self.w_grad = [(layer[layer > 0]).mean() for layer in x]

    def show(self):

        # self.fig.canvas.draw()
        # self.fig.canvas.flush_events()
        plt.ylabel('Mean sum')
        plt.xlabel('Layer')
        plt.show()


class FilterColor:
    def plot_ndarrays(self, filters: [np.ndarray]):
        num_images = len(filters)
        num_cols = 4  # Number of columns for subplots
        num_rows = num_images // num_cols + (1 if num_images % num_cols != 0 else 0)

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 3 * num_rows))

        for i, image in enumerate(filters):
            ax = axes[i // num_cols, i % num_cols] if num_rows > 1 else axes[i % num_cols]
            ax.imshow(image, cmap='gray')
            ax.axis('off')

        # Remove empty subplots if there are any
        for j in range(i + 1, num_rows * num_cols):
            axes.flatten()[j].axis('off')

        # Adjust layout
        plt.tight_layout()
        plt.show()
