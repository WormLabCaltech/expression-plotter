"""
"""

# TODO: entire class is a work in progress
class Experiment:
    """
    Experiment class.
    """

    # metadata
    # TODO: what metadata must be stored?
    title = ''
    path = ''
    date = None
    category = None
    ctrl = None
    threshold = 0.05
    n = 10**5
    ova = False

    p_vals = None
    q_vals = None

    # TODO: add more initializers
    def __init__(self, path, title, **kwargs):
        self.path = path
        self.title = title
        self.date = kwargs.pop('date', None)
        self.category = kwargs.pop('category', None)
        self.ctrl = kwargs.pop('ctrl', None)
        self.threshold = kwargs.pop('threshold', 0.05)
        self.n = kwargs.pop('n', 10**5)
        self.ova = kwargs.pop('ova', False)

    ##
    # TODO: implement the following methods
    ##
    def read_data(self):
        """
        Reads data from the specified path and extracts relevant data.
        """
        pass

    def get_p_matrix(self):
        """
        Retrieves the p value matrix. Calculates the matrix if it has not before.
        """
        return self.p_vals

    def get_q_matrix(self):
        """
        Retrieves the q value matrix. Calculates the matrix if it has not before.
        """
        return self.q_vals

    def save_p_matrix(self, path=self.title):
        """
        Saves the p value matrix to the specified path.
        """
        pass

    def save_q_matrix(self, path=self.title):
        """
        Saves the q value matrix to the specified path.
        """
        pass

    def plot_p_heatmap(self, path=self.title, show=False, **kwargs):
        """
        Saves the p value heatmap to specified path. Optionally shows the plot.
        """
        pass

    def plot_q_heatmap(self, path=self.title, show=False, **kwargs):
        """
        Saves the q value heatmap to specified path. Optionally shows the plot.
        """
        pass

    def plot_boxplot(self, path=self.title, show=False, **kwargs):
        """
        Saves the boxplot to specified path. Optionally shows the plot.
        """
        pass

    def plot_jitterplot(self, path=self.title, show=False, **kwargs):
        """
        Saves the jitterplot to specified path. Optionally shows the plot.
        """
        pass

    def calculate_all(self):
        """
        Calculates the p and q value matrices.
        """
        self.get_p_matrix()
        self.get_q_matrix()
        pass

    def save_all(self):
        """
        Saves the p and q value matrices and all possible plots.
        """
        pass

    def all(self):
        """
        Perform complete analysis. (Calculate p and q value matrices, saves them,
        and saves all possible plots.)
        """
        pass
