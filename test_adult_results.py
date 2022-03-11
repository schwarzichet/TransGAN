import pandas
from sdv.tabular import GaussianCopula
from sdgym.synthesizers import CTGAN, Identity, Uniform, Independent
import sdgym


def gaussian_copula(real_data, metadata):
    gc = GaussianCopula(default_distribution="gaussian")
    table_name = metadata.get_tables()[0]
    gc.fit(real_data[table_name])
    print("this is gc")
    print(gc.sample())
    print(gc.sample().describe())
    return {table_name: gc.sample()}


def transgan(real_data, metadata):
    table_name = metadata.get_tables()[0]
    sample = pandas.read_csv("transgan_gen_adult_data_3.csv")
    print("this is transgan")
    print(sample)
    print(sample.describe())
    # num = sample._get_numeric_data()
    # num[num < 0] = 0
    # print(sample)
    # print(sample.describe())
    return {table_name: sample}


scores = sdgym.run(
    # synthesizers=[gaussian_copula, Identity, Uniform, Independent,CTGAN, transgan],
    synthesizers=[transgan],
    datasets=["adult"],
    workers=4,
    show_progress=True,
    # metrics=['BinaryMLPClassifier'],
    # target='label'
)


print(scores)
