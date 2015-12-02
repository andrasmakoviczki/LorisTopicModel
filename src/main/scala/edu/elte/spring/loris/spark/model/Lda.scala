package edu.elte.spring.loris.spark.model

import scala.collection.mutable
import scala.util.Try

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.clustering.EMLDAOptimizer
import org.apache.spark.mllib.clustering.LDA
import org.apache.spark.mllib.clustering.OnlineLDAOptimizer
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.rdd.RDD
import org.apache.spark.rdd.RDD.rddToPairRDDFunctions

import com.codahale.jerkson.Json.generate
import com.typesafe.config.Config
import com.typesafe.config.ConfigFactory

import edu.elte.spring.loris.spark.model.Lda.Topic
import spark.jobserver.SparkJob
import spark.jobserver.SparkJobValid
import spark.jobserver.SparkJobValidation

object Lda extends SparkJob {

  case class Topic(topicName: String, topicValue: Double) extends Serializable
  case class TopicPair(rowId: String, topic: Array[Topic]) extends Serializable

  def main(args: Array[String]) {
    val conf = new SparkConf()
      .setAppName("LorisLDA")
      .setMaster("local")
    val sc = new SparkContext(conf)

    val config = ConfigFactory.parseString("")
    val results = runJob(sc, config)
  }

  override def validate(sc: SparkContext, config: Config): SparkJobValidation = {
    SparkJobValid
  }

  override def runJob(sc: SparkContext, config: Config): Any = {

    val maxSize: Int = Try(config.getInt("input.maxFeedSize")).getOrElse(100)
    
    //Új FeedEntry-k lekérése
    val feeds = NewFeeds.getFreshFeeds(maxSize)

    val topicList = scala.collection.mutable.ListBuffer.empty[TopicPair]
    
    for (f <- feeds) {
      //LDA kiszámítása
      val topics: Array[Topic] = LDACompute(f, sc,config)
      val topicPair: TopicPair = TopicPair(f.rowid, topics)
      topicList += topicPair
    }
    val elapsedSeconds = (System.nanoTime() - start) / 1e9
    println("elapsed time: " + elapsedSeconds)

    //JSON formátum
    return generate(topicList)
  }

   def LDACompute(feed: FeedEntry, sc: SparkContext, config: Config): Array[Topic] = {

    val stopWordInput: String = Try(config.getString("input.stopWord")).getOrElse("hun_stop.txt")
    val minWordLength: Int = Try(config.getInt("input.minWordLength")).getOrElse(3)

    val algorithm: String = Try(config.getString("input.algorithm")).getOrElse("em")
    val k: Int = Try(config.getInt("input.k")).getOrElse(10)
    val maxIteration: Int = Try(config.getInt("input.maxIteration")).getOrElse(10)

    //Adatok előkészítése
    val (documents, vocabArray) = preprocess(sc, feed, stopWordInput, minWordLength)

    if (documents == null){
       return new Array[Topic](0)
    }
    
    val lda = new LDA()

    val optimizer = algorithm match {
      case "em" => new EMLDAOptimizer()
      case "online" => new OnlineLDAOptimizer()
    }

    lda.setOptimizer(optimizer).setK(k).setMaxIterations(maxIteration)

    //Lda számítás
    val ldaModel = lda.run(documents)
    val topicIndeces = ldaModel.describeTopics(maxTermsPerTopic = 10)

    //Sorszámok helyettesítése szavakkal
    val topics = topicIndeces.map {
      case (terms, termWeights) =>
        terms.zip(termWeights).map {
          case (term, weight) =>
            (vocabArray(term.toInt), weight)
        }
    }

    //Legmagasabb pontszámú szavakkal való visszatérés
    return sc.parallelize(topics.flatten.toSeq)
      .groupByKey()
      .map(x => new Topic(x._1, x._2.max))
      .sortBy(-_.topicValue)
      .take(10)
  }

   def preprocess(sc: SparkContext, feed: FeedEntry, stopWordInput: String, minWordLength: Int): (RDD[(Long, Vector)], Array[String]) = {
    val corpus = feed.title.concat(" ".concat(feed.content))
    val stopwordText = sc.textFile(stopWordInput).collect().flatMap(_.stripMargin.split("\\s+")).toSet

    //Tokenizálás és tisztítás
    val tokenized : RDD[Seq[String]] =  
    sc.parallelize(Seq(corpus.toLowerCase.split("\\s+").filter(_.forall(java.lang.Character.isLetter)).filter(!stopwordText.contains(_)).filter(_.length > minWordLength)))
         
    if (tokenized.collect.flatten.size == 0){
       return (null,null)
    }

    //Szótár készítése
    val wordcount: Array[(String, Long)] =
      tokenized.flatMap(_.map(_ -> 1L)).reduceByKey(_ + _).collect
    val fullVocabSize = wordcount.size
    val vocabArray: Array[String] =
      wordcount.map(_._1)
    
    //Szótár indexelése
    val vocab: Map[String, Int] = vocabArray.zipWithIndex.toMap

    //Dokumentum elkészítése
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
          //Sorszám, vektor előállítása
          (id, Vectors.sparse(vocab.size, wc.toSeq))
      }
    return (documents, vocabArray)
  }

}