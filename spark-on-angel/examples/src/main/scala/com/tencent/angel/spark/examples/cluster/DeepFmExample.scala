package com.tencent.angel.spark.examples.cluster

import com.tencent.angel.RunningMode
import com.tencent.angel.conf.AngelConf
import com.tencent.angel.ml.core.conf.SharedConf
import com.tencent.angel.spark.context.PSContext
import com.tencent.angel.spark.ml.core.ArgsUtil
import com.tencent.angel.spark.ml.util.DataLoader
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.storage.StorageLevel

class DeepFmExample {

  def main(args: Array[String]): Unit = {
    val params = ArgsUtil.parse(args)
    val input = params.getOrElse("input", "")
    val output = params.getOrElse("output", "")
    val actionType = params.getOrElse("actionType", "train")
    val network = params.getOrElse("network", "LogisticRegression")
    val modelPath = params.getOrElse("model", "")

    SharedConf.get().set(AngelConf.ANGEL_RUNNING_MODE, RunningMode.ANGEL_PS.toString)
    SharedConf.addMap(params)

    val dim = SharedConf.indexRange.toInt

    println(s"dim=$dim")


    val conf = new SparkConf()

    if (modelPath.length > 0)
      conf.set(AngelConf.ANGEL_LOAD_MODEL_PATH, modelPath + "/back")

    val sc = new SparkContext(conf)

    PSContext.getOrCreate(sc)

    val className = "com.tencent.angel.spark.ml.classification." + network



    val data = sc.textFile(input).filter(f => f.length > 0 && f != null)
      .map(s => (DataLoader.parseLongDummy(s, dim), DataLoader.parseLabel(s, false)))
      .map {
        f =>
          f._1.setY(f._2)
          f._1
      }.filter(f => f != null).filter(f => f.getX.getSize > 0)

    data.persist(StorageLevel.DISK_ONLY)


  }

}
