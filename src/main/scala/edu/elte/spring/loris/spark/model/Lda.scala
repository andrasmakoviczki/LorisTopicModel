package edu.elte.spring.loris.spark.model

import org.apache.spark.{ SparkContext, SparkConf }
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.{ Vector, Vectors }
import org.apache.spark.mllib.clustering.{ EMLDAOptimizer, OnlineLDAOptimizer, DistributedLDAModel, LDA }

import scala.collection.mutable

object Lda {
  def main(args: Array[String]) {

    val conf = new SparkConf()
      .setAppName("LDA")
      .setMaster("local")
    val sc = new SparkContext(conf)

    import org.apache.spark.rdd.RDD

    val inputPath: String = "text2.txt"
    val stopWordInput: String = "hun_stop.txt"

    val k: Int = 10
    val maxIteration: Int = 10
    var algorithm: String = "em"

    val corpus: RDD[String] = sc.wholeTextFiles(inputPath).map(_._2)
    val stopwordText = sc.textFile(stopWordInput).collect().flatMap(_.stripMargin.split("\\s+")).toSet

    val minWordLength = 3

    //tokenizálás
    val tokenized: RDD[Seq[String]] =
      corpus.map(_.toLowerCase.split("\\s+")).map(_.filter(_.forall(java.lang.Character.isLetter)).filter(!stopwordText.contains(_)).filter(_.length > minWordLength))

    //wordcount: sort
    val wordcount: Array[(String, Long)] =
      tokenized.flatMap(_.map(_ -> 1L)).reduceByKey(_ + _).collect

    val fullVocabSize = wordcount.size
    //méretezhető szótár

    val vocabArray: Array[String] =
      wordcount.map(_._1)
    val vocab: Map[String, Int] = vocabArray.zipWithIndex.toMap

    val documents: RDD[(Long, Vector)] =
      tokenized.zipWithIndex.map {
        case (tokens, id) =>
          val wc = new mutable.HashMap[Int, Double]()
          tokens.foreach {
            term =>
              if (vocab.contains(term)) {
                val termIndex = vocab(term)
                wc(termIndex) = wc.getOrElse(termIndex, 0.0) + 1.0
              }
          }
          //kulcsok rendezése?
          (id, Vectors.sparse(vocab.size, wc.toSeq))
      }

    //
    algorithm = "em"
    val lda = new LDA()

    val optimizer = algorithm match {
      case "em" => new EMLDAOptimizer()
      case "online" => new OnlineLDAOptimizer()
        //mire jó?
        //.setMiniBatchFraction(0.05)
    }

    lda.setOptimizer(optimizer).setK(k).setMaxIterations(maxIteration)
    //mire jó?
    //.setDocConcentration(docConcentration)
    //.setTopicConcentration(topicConcentration)

    val ldaModel = lda.run(documents)

    /*
      if (ldaModel.isInstanceOf[DistributedLDAModel]) {
      val distLDAModel = ldaModel.asInstanceOf[DistributedLDAModel]
      val avgLogLikelihood = distLDAModel.logLikelihood / actualCorpusSize.toDouble
      println(s"\t Training data average log likelihood: $avgLogLikelihood")
      println()
    }
      */

    //Array[(Array[Int], Array[Double])]
    val topicIndeces = ldaModel.describeTopics(maxTermsPerTopic = 10)

    val topics = topicIndeces.map {
      case (terms, termWeights) =>
        terms.zip(termWeights).map {
          case (term, weight) =>
            (vocabArray(term.toInt), weight)
        }
    }
    
    topics.flatten.sortBy(-_._2).map(_._1).distinct
    
    topics.flatten.sortBy(-_._2).take(10).foreach{
      case (topic,weight) =>
        println(s"$topic\t$weight")
    }


    //termPerTopic indexeivel összefűzés
    topics.zipWithIndex.foreach {
      case (topic, i) =>
        println(s"TOPIC $i")
        topic.foreach {
          case (term, weight) =>
            println(s"$term\t$weight")
        }
        println
    }
  }

  private def run() {

  }
}