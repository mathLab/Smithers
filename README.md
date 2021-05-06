**SMITHERS**: Smithers' a Mathematical Interdisciplinary Toolbox for Helping Engineers Researchers and Scientist

## Description
**Smithers** is a generic library for scientific computing developed in Python
that aims to facilitate the development of many typical routines. It is
substantially a multi-purpose toolbox that inherits functionality from other
packages to make easier and compact the coding of recurrent workflows.



## Dependencies and installation
**Smithers** requires `numpy`, `scipy` and `matplotlib`, which are the common libraries for scientific computing in the Python community.
All the other dependencies of **Smithers** are imported *at runtime*. 



### Installing from source
The official distribution is on GitHub, and you can clone the repository using
```bash
> git clone https://github.com/mathLab/Smithers
```

To install the package just type:
```bash
> python setup.py install
```

To uninstall the package you have to rerun the installation and record the installed files in order to remove them:

```bash
> python setup.py install --record installed_files.txt
> cat installed_files.txt | xargs rm -rf
```

## Testing

We are using Travis CI for continuous intergration testing. You can check out the current status [here](https://travis-ci.org/mathLab/PyDMD).

To run tests locally (`nose` is required):

```bash
> python test.py
```

## License

See the [LICENSE](LICENSE.rst) file for license rights and limitations (MIT).
