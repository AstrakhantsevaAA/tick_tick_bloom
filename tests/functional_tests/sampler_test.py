from src.train.classificator.sampler import define_sampler


def test_define_sampler(dataset):
    define_sampler(dataset)
