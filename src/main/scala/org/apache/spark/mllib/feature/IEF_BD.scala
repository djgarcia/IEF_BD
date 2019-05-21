package org.apache.spark.mllib.feature

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, LabeledPoint => NewLabeledPoint}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

class IEF_BD(val data: RDD[LabeledPoint], k: Int = 4, threshold: Double = 0.5, maxIterations: Int = 3, nTrees: Int = 200, maxDepth: Int = 10, seed: Int = 48151623, sqlContext: SparkSession) extends Serializable {

  import sqlContext.implicits._

  def runFilter(): RDD[LabeledPoint] = {

    val maxError = k * threshold
    var iterations = 0
    var finalData = data

    while (iterations < maxIterations) {

      var predictions = finalData.map { l =>
        (l, new Array[Double](k))
      }.repartition(1024).persist()

      val cvdat = MLUtils.kFold(predictions.map(_._1), k, seed)

      for (i <- cvdat.indices) {

        val trainAsML = cvdat(i)._1.map(e => NewLabeledPoint(e.label, e.features.asML)).toDS().persist()

        val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").setHandleInvalid("keep").fit(trainAsML)

        val rf = new RandomForestClassifier().setLabelCol("indexedLabel").setFeaturesCol("features").setNumTrees(nTrees).setMaxDepth(maxDepth).setSeed(seed)

        val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)

        val pipeline = new Pipeline().setStages(Array(labelIndexer, rf, labelConverter))

        val model = pipeline.fit(trainAsML)

        val predictionsAsML = predictions.map(e => NewLabeledPoint(e._1.label, e._1.features.asML)).toDS().persist()

        val predsAndLabels = model.transform(predictionsAsML).rdd.map(row => (
          row.getAs[String]("predictedLabel"),
          row.getAs[Double]("label"),
          row.getAs[org.apache.spark.ml.linalg.Vector]("features")
        )).persist()

        val joinedPreds = predsAndLabels.zipWithIndex().map(_.swap).join(predictions.map(_._2).zipWithIndex().map(_.swap)).map(_._2).persist()

        predictions = joinedPreds.map { l =>
          val featuresAndLabel = LabeledPoint(l._1._2, Vectors.dense(l._1._3.toArray))
          val predArray = l._2
          predArray(i) = l._1._1.toDouble
          (featuresAndLabel, predArray)
        }.persist()
      }

      finalData = predictions.map { l =>
        val label = l._1.label
        val error = l._2.map { i =>
          if (i == label)
            0
          else
            1
        }.sum
        if (error >= maxError)
          LabeledPoint(-1, l._1.features)
        else
          l._1
      }.filter { point => point.label != -1 }

      iterations += 1

      println("Iteration: " + iterations)
      println("Deleted: " + (data.count - finalData.count))
    }
    finalData
  }
}
