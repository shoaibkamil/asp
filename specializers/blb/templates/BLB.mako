import java.util.Random
import spark.SparkContext
import SparkContext._

//class BLB(numSubsamples: Int = 100, numBootstraps: Int = 25, subsampleLenExp: Double = 0.5) {
//  // Maybe I should change these so that they return a list of indices
//  private def subsample(data: Array[Double], subsampleLenExp: Double): Array[Double] = {
//    val subsampleLen = math.pow(data.length, subsampleLenExp).toInt
//    val rand = new Random()
//    var subsample = new Array[Double](subsampleLen)
//    // Sample without replacement from Knuth
//    var i = 0
//    var j = 0
//    while (j < subsampleLen) {
//      if (rand.nextDouble() * (data.length - i) > subsampleLen - j) {
//        i += 1
//      } else {
//        subsample(j) = data(i)
//        i += 1
//        j += 1
//      }
//    }
//    return subsample
//  }
//
//  private def bootstrap(data: Array[Double]): Array[Double] = {
//    var bootstrap = new Array[Double](data.length)
//    val rand = new Random()
//    for (i <- 0 until data.length) {
//      bootstrap(i) = data(rand.nextInt(data.length))
//    }
//    return bootstrap
//  }
//
//  // These are user-defined
//  def computeEstimate(arr: Array[Double]): Double = {
//    return arr.reduce(_+_)/arr.length
//  }
//
//  def reduceBootstraps(arr: Array[Double]): Double = {
//    return arr.reduce(_+_)/arr.length
//  }
//
//  def average(arr: Array[Double]): Double = {
//    return arr.reduce(_+_)/arr.length
//  }
//
//  def run(data: Array[Double]): Double = {
//    var subsampleEstimates = new Array[Double](numSubsamples)
//    for (i <- 0 until numSubsamples) {
//      val subsampleSet = subsample(data, subsampleLenExp)
//      var bootstrapEstimates = new Array[Double](numBootstraps)
//      for (j <- 0 until numBootstraps) {
//        val bootstrapSet = bootstrap(subsampleSet)
//        val estimate = computeEstimate(bootstrapSet)
//        bootstrapEstimates(j) = estimate
//      }
//      subsampleEstimates(i) = reduceBootstraps(bootstrapEstimates)
//    }
//    return average(subsampleEstimates)
//  }
//}

object BLB {
  private def subsample(data: Array[Double], subsampleLen: Int, seed: Int): Array[Double] = {
    val rand = new Random(seed)
    var subsample = new Array[Double](subsampleLen)
    // Sample without replacement from Knuth
    var i = 0
    var j = 0
    while (j < subsampleLen) {
      if (rand.nextDouble() * (data.length - i) > subsampleLen - j) {
        i += 1
      } else {
        subsample(j) = data(i)
        i += 1
        j += 1
      }
    }
    return subsample
  }

  private def bootstrap(data: Array[Double], seed: Int): Array[Double] = {
    var bootstrap = new Array[Double](data.length)
    val rand = new Random(seed)
    for (i <- 0 until data.length) {
      bootstrap(i) = data(rand.nextInt(data.length))
    }
    return bootstrap
  }

  private def computeEstimate(arr: Array[Double]): Double = {
    ${computeEstimate}
  }

  private def reduceBootstraps(arr: Array[Double]): Double = {
    ${reduceBootstraps}
  }

  private def average(arr: Array[Double]): Double = {
    ${average}
  }

  def run(data: Array[Double], sc: SparkContext, seed: Int = 0, numSubsamples: Int = 100, numBootstraps: Int = 25, subsampleLenExp: Double = 0.5): Double = {
    val subsampleLen = math.pow(data.length, subsampleLenExp).toInt
    val subsamples = (0 until numSubsamples).map( i => {
      subsample(data, subsampleLen, seed + i)
    })
    val subsampleResults = sc.parallelize(subsamples).map(subsample => {
        val estimates = (0 until numBootstraps).map( j => {
            computeEstimate(bootstrap(subsample, seed + j))
        })
        reduceBootstraps(estimates.toArray)
    })
    return average(subsampleResults.toArray)
  }

  def main(args: Array[String]) {
    //val data = Range.Double(0. , 10000., 1.).toArray
    var arr: List[Double] = scala.util.parsing.json.JSON.parse(args(0)).getOrElse(List()) match {
      case x:List[Double] => x
    }
    val data = arr.toArray
    val sc = new SparkContext("local", "blb")
    println(run(data, sc, new Random().nextInt()))
  }
}
