import multiprocessing

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import utils
from model import feature_selection_with_nsga2, get_toolbox

path = './'  # the path of input and output data files
matplotlib.use('Agg')


def preprocess_dataset(filename, col_headings, column_name_class='', column_list_drop=[]):
    # preprocess the dataset
    df = pd.read_csv(path + filename, na_filter=True)
    df.columns = col_headings  # add headings

    if column_name_class != '':
        class_column = df.pop(column_name_class)  # Remove the column from the DataFrame and store it
        # Insert the column at the end of the dataframe column-wise
        df['class'] = class_column
        # df.insert(new_position, column_name, column)

    # drop the undesired column
    if len(column_list_drop) != 0:
        for _ in column_list_drop:
            df = df.drop(_, axis=1)

    # Assign 1 to the string 'M' in column 'M'
    df.loc[df[column_name_class] == 'M', column_name_class] = 1
    df.loc[df[column_name_class] == 'B', column_name_class] = 0

    df.to_csv('./' + filename[:filename.find('.')] + '.csv', index=False)
    print(df.head)
    print('Size of dataset:', filename, df.shape)


def main():
    # import dataset
    file_name = 'WBCD.data'
    # construct headings
    features = ['V' + str(i) for i in range(1, 31)]
    col_head = ['idx', 'class']
    col_head.extend(features)
    column_name_class = 'class'  # the class label column
    column_list_drop = ['idx']
    preprocess_dataset(file_name, col_head, column_name_class, column_list_drop)

    # NSGA-II
    model_name = '5NN'
    dataset_name = 'WBCD'
    setting = 0.3

    with multiprocessing.Pool() as pool:
        toolbox = get_toolbox(model_name, dataset_name, setting, pool)
        gens, _ = feature_selection_with_nsga2(toolbox,
                                               num_generation=100,
                                               num_population=100,
                                               crossover_prob=0.9,
                                               mutate_prob=0.01)
    pop = utils.sort_population(gens[-1])
    accuracy = pop[0].fitness.values[0]
    dim_reduction = pop[0].fitness.values[1]
    print('Dimension reduction: {:.2%}'.format(dim_reduction))
    print("Accuracy: {:.2%}\n".format(accuracy))

    front = np.asarray([ind.fitness.values for ind in pop])
    plt.title('Optimal front derived with NSGA-II', fontsize=12)

    # The map() function takes two arguments: a function and an iterable.
    # It applies the given function to each element of the iterable and returns an iterator that produces the results.
    accuracy = list(map(lambda x: x[0], front))
    dim_reduction = list(map(lambda x: x[1], front))
    plt.scatter(dim_reduction, accuracy, c="b")
    plt.scatter(dim_reduction[0], accuracy[0], c="r")

    plt.xlabel('DR(%)')
    plt.ylabel('Accuracy(%)')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
