# IEF_BD

This method implements an Iterative Ensemble Filter for noise filtering in Big Data (IEF_BD).

This software has been proved with seven large real-world datasets such as:
- skin_noskin dataset: 245K instances and 3 attributes. https://archive.ics.uci.edu/ml/datasets/skin+segmentation
- ht_sensor dataset: 929K instances and 11 attributes. http://archive.ics.uci.edu/ml/datasets/gas+sensors+for+home+activity+monitoring
- watch_acc dataset: 3,5M instances and 20 attributes. https://archive.ics.uci.edu/ml/datasets/Heterogeneity+Activity+Recognition
- watch_gyr dataset: 3,2M instances and 20 attributes. https://archive.ics.uci.edu/ml/datasets/Heterogeneity+Activity+Recognition
- SUSY dataset: 5M instances and 18 attributes. https://archive.ics.uci.edu/ml/datasets/SUSY
- HIGGS dataset: 11M instances and 28 attributes. https://archive.ics.uci.edu/ml/datasets/HIGGS

## Brief benchmark results:
* IEF_BD has shown to be the best performing noise filter algorithm, achieving the best accuracy.
* IEF_BD can outperform the current best noise filter for Big Data, HME-BD, for all datasets and levels of noise.
* IEF_BD has also proved to be an effective solution for transforming raw Big Data into Smart Data.

## Example (IEF_BD)

```scala
import org.apache.spark.mllib.feature._

val partitions = 4
val threshold = 0.75
val maxIterations = 3
val nTrees = 200
val maxDepth = 12

// Data must be cached in order to improve the performance

val ief_bd_model = new IEF_BD(trainingData, // RDD[LabeledPoint]
                              partitions, // number of partitions
                              threshold, // threshold for the vote strategy
                              maxIterations, // number of iterations
                              nTrees, // size of the Random Forests
                              maxDepth, // depth of the Random Forests
                              seed) // seed for the Random Forests

val ief_bd = ief_bd_model.runFilter()
```
