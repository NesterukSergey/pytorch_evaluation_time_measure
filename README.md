This code evaluates the time of evaluation of Pytorch model on the example of MobileNetv2. To avoid caching data and exclude time of IO operations, the input data is generated randomly on each iteration. 


To run code:

```git clone https://github.com/NesterukSergey/pytorch_evaluation_time_measure.git```

```cd pytorch_evaluation_time_measure```

```pip install -r requirements.txt```

```python ./timetest_pipeline.py ```


All the settings are in the *config.json* file.

*batch_sizes* parameter sets the maximum batch size (powers of 2). Not recommended to set greater than 6.
