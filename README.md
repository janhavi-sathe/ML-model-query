
## Installation
We recommend you use `conda` environment. You can set up the virtual environment with following command:
```conda env create -f conda_env.yml ```

gym 0.21 installation is broken with recent versions of setuptools and wheel (ref: https://stackoverflow.com/a/77205046).
Please downgrade them to the following versions:
```
    pip install setuptools==65.5.1 pip==21 wheel==0.38.0
```

Then, please install `core` and `domains` packages as follows:
```
  pip install -e core/
  pip install -e domains/
```

## Execution
* Please read the README files in `domains/` and `web_app_v2/` directory.


## Development

### Style Guide
This project uses a custom Python style guide, which differs from [PEP 8](https://www.python.org/dev/peps/pep-0008/) in the following ways:
- Use two-space indentation instead of four-space indentation.
- 80 character line limits rather than 79.

You can format your code according to the style guide using linters (e.g., [flake8](https://pypi.org/project/flake8/)) and autoformatters (e.g., [yapf](https://github.com/google/yapf)).
