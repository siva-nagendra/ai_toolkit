import os
import time
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class DataVisualizer:
    """
    A class used to visualize data attributes like file type distribution,
    file size distribution, embedding time per document, and documents processed over time.

    Methods
    -------
    start_timer():
        Initializes the timer.
    record_file_info(documents: list):
        Records information about the file such as extensions and sizes.
    generate_plots(documents: list, save_path: str):
        Generates and saves a collection of plots visualizing the data.
    """

    def __init__(self):
        """
        Initializes the DataVisualizer instance with required attributes.
        """
        self.file_extensions = []
        self.file_sizes = []
        self.embedding_times = []
        self.doc_count = [0]  # initialize with 0
        self.times = [0]  # initialize with 0
        self.start_time = None
        logging.info("DataVisualizer instance created.")

    def start_timer(self):
        """
        Initializes the timer by recording the current time.
        """
        self.start_time = time.time()
        logging.info("Timer started.")

    def record_file_info(self, documents):
        """
        Records the file extensions and sizes from the provided list of documents.

        Parameters:
        documents (list): A list of document file paths.
        """
        if not documents:
            logging.warning("No documents provided to record_file_info method.")
            return

        plot_data = {}
        embedding_time = time.time() - self.start_time

        self.file_extensions = [os.path.splitext(f)[1] for f in documents]
        self.file_sizes = []
        for f in documents:
            if os.path.exists(f):
                self.file_sizes.append(os.path.getsize(f))
                self.embedding_times.append(embedding_time)
            else:
                logging.error(f"File not found: {f}")

        self.doc_count.append(len(documents))
        self.times.append(time.time() - self.start_time)

    def generate_plots(self, documents=[], save_path='enhanced_progress_plot.png'):
        """
        Generates and saves a collection of plots visualizing the data.

        Parameters:
        documents (list): A list of document file paths.
        save_path (str): The path where the plot image should be saved.
        """
        self.record_file_info(documents)

        file_extension_counts = Counter(self.file_extensions)
        file_size_bins = np.linspace(min(self.file_sizes), max(self.file_sizes), 10)

        fig, axs = plt.subplots(2, 2, figsize=(15, 15))

        # Plot 1: Document Type Distribution
        axs[0, 0].bar(file_extension_counts.keys(), file_extension_counts.values())
        axs[0, 0].set_title('Document Type Distribution')
        axs[0, 0].set_xlabel('File Extension')
        axs[0, 0].set_ylabel('Count')

        # Plot 2: Document Size Distribution
        axs[0, 1].hist(self.file_sizes, bins=file_size_bins, edgecolor='black')
        axs[0, 1].set_title('Document Size Distribution')
        axs[0, 1].set_xlabel('File Size (bytes)')
        axs[0, 1].set_ylabel('Count')

        # Plot 3: Embedding Time per Document
        axs[1, 0].plot(self.embedding_times)
        axs[1, 0].set_title('Embedding Time per Document')
        axs[1, 0].set_xlabel('Document Index')
        axs[1, 0].set_ylabel('Time (s)')

        # Plot 4: Documents Processed Over Time
        axs[1, 1].plot(self.times, self.doc_count)
        axs[1, 1].set_title('Documents Processed Over Time')
        axs[1, 1].set_xlabel('Time (s)')
        axs[1, 1].set_ylabel('Number of Documents Processed')

        plt.tight_layout()
        plt.savefig(save_path)
        logging.info(f"Enhanced progress plot saved as '{save_path}'")
