# TransGAN for tabular data

code to immigrate TransGAN for tabular data

## install

install transgan's dependency

install sdv from https://github.com/sdv-dev/CTGAN

install SDGym from https://github.com/sdv-dev/SDGym

install rtdl from https://github.com/Yura52/rtdl


## Before Testing
go to rtdl's `modules.py` find `class CategoricalFeatureTokenizer` replace its `forward` methods as
```
def forward(self, x: Tensor) -> Tensor:
    if x[0][0].dtype == torch.int:
        x = self.embeddings(x + self.category_offsets[None])
    else:
        x = x@self.embeddings.weight

    if self.bias is not None:
        x = x + self.bias[None]
    return x
```

## Training
python exp/ft_train_adult.py

## Generate samples
python exp/ft_test_adult.py

## Test results
python test_adult_results.py
