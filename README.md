# Friendly Guacamole :avocado:
Microbiome-friendly scikit-learn style data preprocessing and other utilities.

friendly-guacamole is designed to integrate common microbiome datatypes and 
transformations with scikit-learn transformers and models.
## Example

```python
from friendly_guacamole.datasets import KeyboardDataset
from friendly_guacamole.transforms import AsDense, CLR
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

dataset = KeyboardDataset('some/relative/path')

pipeline = Pipeline([
    ('convert_type', AsDense()),
    ('clr', CLR()),
    ('pca', PCA()),
])

pipeline.fit_transform(dataset['table'])
```