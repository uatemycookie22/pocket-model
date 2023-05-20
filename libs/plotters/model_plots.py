import time
import matplotlib.pyplot as plt


class CostRT:
    def __init__(self):
        self.costs = []
        self.accuracies = []
        self.fig = None
        self.ax = None
        self.line1 = None
        self.line2 = None
        self.timestamps = []
        self.start = None

    def plot(self):
        plt.ion()
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.line1, = self.ax.plot([], [], 'r-')  # Returns a tuple of line objects, thus the comma
        self.line2, = self.ax.plot([], [], 'g-')  # Returns a tuple of line objects, thus the comma

        self.start = time.time()

    def add(self, x, x2):
        self.costs.append(x)
        self.accuracies.append(x2)
        self.timestamps.append(time.time() - self.start)

    def show(self):
        self.line1.set_xdata(self.timestamps)
        self.line1.set_ydata(self.costs)
        self.line2.set_xdata(self.timestamps)
        self.line2.set_ydata(self.accuracies)
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.ylabel('Cost (absolute)')
        plt.xlabel('Time (seconds)')
        #plt.legend(loc="best")
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
        #examples.sort(key=lambda x: x['cost'])

        # Get the top 5 examples
        low_5 = [example for example in examples if example['prediction'] == example['label']]
        low_5 = low_5[len(low_5)//2 + 2000:len(low_5)//2 + 15 + 2000]

        # Prepare the subplots
        fig, axes = plt.subplots(1, 15, figsize=(15, 3))

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
