Example of searching betweeen datasets

```
>>> from smithers.dataset import DatasetCollector
>>> collector = DatasetCollector()
>>> avail_datasets = collector.search()
>>> print(avail_datasets)
```



Example of using one dataset 

```
>>> from smithers.dataset import NavierStokesDataset
>>> ns = NavierStokesDataset()
>>> print(ns.snapshots)
>>> print(ns.snapshots['p'].shape)
>>> ns.plot(out='p')
```