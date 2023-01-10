import random
from typing import List


class ITree:
    """
    A class for building Isolation Trees.
    """

    @staticmethod
    def build(samples: List[List[float]], max_depth: int, curr_depth: int = 0) -> dict:
        """
        Build a isolation tree from given samples, recursively.
        The building is done in the following way:
            1. If max_depth was is reached or no more samples, create and return an external node which will "contain"
               all the current samples
            2. Pick a random non constant attribute from the samples attributes.
            3. If all of the attributes are constant, create and return an external node with all current samples.
            3. Pick a random split value between the min and max values for the selected attributes among the samples
            4. Create a left node recursively with all the samples that their selected attribute value is less
               than the chosen split value
            5. Create a right node with all the samples that their selected attribute value is greater or
               equal to the chosen split value, recursively.
            6. Create and return an internal node and set the split_attribute_idx, split_value, attr_min_val,
               attr_max_val, left_node and right_node accordingly.
        :param samples: The samples to train (build) the tree with
        :param max_depth: The max allowed depth of the tree. The build will stop adding nodes to a branch if
            the max_depth is reached, and will add all the nodes that reached there as an external node
        :param curr_depth: The current depth in the tree
        :return: Fully trained isolation tree (in form of a dictionary)
        """
        # TODO: Validate params
        num_samples = len(samples)

        if curr_depth >= max_depth or num_samples <= 1:
            return {"type": "external", "num_samples": num_samples}

        # Calculate the max value of each attribute of among the given samples
        max_in_columns = ITree.matrix_col_max(samples)

        # Calculate the min value of each attribute of among the given samples
        min_in_columns = ITree.matrix_col_min(samples)

        zipped_min_max = zip(range(num_samples), min_in_columns, max_in_columns)

        # Filter attribute indices where the min and max is the same
        filtered_zipped_min_max = [
            (i, min_val, max_val) for (i, min_val, max_val) in zipped_min_max if min_val != max_val
        ]

        if not filtered_zipped_min_max:
            # No non-constant attributes left, aka all samples are the same
            # Therefore reached external node
            return {"type": "external", "num_samples": num_samples}

        # Randomly select a non constant attribute
        split_attribute_idx, attr_min_val, attr_max_val = random.choice(filtered_zipped_min_max)

        # Choose a random split value
        split_val = random.uniform(attr_min_val, attr_max_val)

        left_node_samples = []
        right_node_samples = []

        # Split the samples to right and left nodes
        for sample in samples:
            if sample[split_attribute_idx] < split_val:
                left_node_samples.append(sample)

            else:
                right_node_samples.append(sample)

        left_node = ITree.build(left_node_samples, max_depth, curr_depth + 1)
        right_node = ITree.build(right_node_samples, max_depth, curr_depth + 1)

        return {
            "type": "internal",
            "split_attribute_idx": split_attribute_idx,
            "split_value": split_val,
            "attr_min_val": attr_min_val,
            "attr_max_val": attr_max_val,
            "left_node": left_node,
            "right_node": right_node
        }

    @staticmethod
    def matrix_col_max(matrix: List[List[float]]) -> List[float]:
        """
        Calculates the max value for each column of the matrix
        :param matrix: The matrix of floats
        :return: List of maximum values per column
        """
        return list(map(max, zip(*matrix)))

    @staticmethod
    def matrix_col_min(matrix: List[List[float]]) -> List[float]:
        """
        Calculates the min value for each column of the matrix
        :param matrix: The matrix of floats
        :return: List of minimum values per column
        """
        return list(map(min, zip(*matrix)))


